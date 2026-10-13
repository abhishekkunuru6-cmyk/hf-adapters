# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spyre test for the XPress speculative-decoding DRAFT adapter (``hf_xpress_qwen3``).

Loads the real drafter checkpoint (vocab weights from its target), moves it to
the card, and runs ``propose_block`` against a synthetic target context. Checks
against the fp32 CPU oracle (``tests/_xpress_reference.py``, transcribed from
the speculators training code):

- block hidden states and base logits: per-row cosine >= ``MIN_COSINE``;
- proposals: finite, in-vocab, ``block_size - 1`` of them; top-1 agreement with
  the oracle is *reported*, not asserted — refinement amplifies near-ties and a
  synthetic context has many (the real-context acceptance check is P0/P2).

The target embedding is read from the target checkpoint in the HF cache
(``load_target_embedding`` reads only its shard), or from the safetensors file
in ``XPRESS_TARGET_EMBED`` if set.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_xpress_draft_spyre.py
"""

import os
import time

import pytest
import torch
from safetensors.torch import load_file

from tests._xpress_reference import reference_block_hidden, reference_jacobi
from tests.model_registry import CAUSAL_LM_MODELS, XPRESS_PATHS

MIN_COSINE = 0.99
CTX_LEN = 40
ANCHOR_ID = 1000


def _target_embedding(module, ckpt: str) -> torch.Tensor:
    override = os.environ.get("XPRESS_TARGET_EMBED")
    if override:
        (tensor,) = load_file(override).values()
        return tensor
    target = next(i["target"] for i in CAUSAL_LM_MODELS.values() if i["path"] == ckpt)
    return module.load_target_embedding(target)


def _context(model, start: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    width = len(model.target_layer_ids) * model.config.hidden_size
    positions = start + torch.arange(CTX_LEN)
    return torch.randn(1, CTX_LEN, width, generator=g), positions


def _min_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).min()
    )


# Position 4000 exercises RoPE far from 0, where a reduced-precision inv_freq
# would drift the rotation angles by radians.
@pytest.mark.parametrize("start", [0, 4000], ids=["pos0", "pos4000"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("ckpt", XPRESS_PATHS)
@torch.no_grad()
def test_xpress_propose_block(ckpt, dtype, start):
    import torch_spyre  # noqa: F401

    from hf_adapters import hf_common
    from hf_adapters._xpress_refiner import canonical_head_weights
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    # hf_common.DEVICE (spyre:$LOCAL_RANK) is bound into the adapters at import;
    # select the same card instead of overriding it.
    torch.spyre.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    module = resolve_adapter_module(ckpt)
    assert module.__name__.endswith("hf_xpress_qwen3"), module.__name__

    embed = _target_embedding(module, ckpt)
    config, raw = module.read_checkpoint(ckpt)  # read once, for both models
    oracle = module.build_model(config, raw, embed, dtype=torch.float32)
    feats, positions = _context(oracle, start)
    ctx = slice(CTX_LEN - min(CTX_LEN, module.CTX_PAD), CTX_LEN)
    ref_hidden = reference_block_hidden(
        oracle, feats[:, ctx], positions[ctx], ANCHOR_ID
    )
    ref_base = oracle.lm_head(ref_hidden)
    head_w = {k: v.float() for k, v in canonical_head_weights(raw).items()}
    ref_draft = reference_jacobi(
        head_w, ref_base, ref_hidden, ANCHOR_ID, ANCHOR_ID, oracle.num_passes
    )[0]
    del oracle, head_w

    model = module.build_model(config, raw, embed, dtype=dtype)
    del raw
    hf_common.move_model_to_spyre(model, module, dtype)

    hidden = module.run_draft_block(model, feats, positions, ANCHOR_ID)
    vocab = model.config.vocab_size
    base = model._spyre_xpress_fns["lm_head"](hidden).to("cpu").float()[..., :vocab]
    hidden_cos = _min_row_cosine(hidden.to("cpu"), ref_hidden)
    base_cos = _min_row_cosine(base, ref_base)

    timings = {}
    for topc in (model.topc, 0):
        module.propose_block(model, feats, positions, ANCHOR_ID, topc=topc)  # warm up
        start = time.perf_counter()
        draft = module.propose_block(model, feats, positions, ANCHOR_ID, topc=topc)
        timings[topc] = time.perf_counter() - start
        assert draft.shape == (model.block_size - 1,)
        assert bool(((draft >= 0) & (draft < vocab)).all())
        if topc == 0:
            agree = int((draft == ref_draft).sum())

    print(
        f"\n[xpress {dtype} pos{start}+] hidden cos={hidden_cos:.5f} base-logit cos={base_cos:.5f} "
        f"draft agreement (topc=0 vs fp32 oracle) {agree}/{model.block_size - 1} "
        f"propose latency: "
        + ", ".join(f"topc={c}: {t * 1e3:.1f} ms" for c, t in timings.items())
    )
    assert torch.isfinite(hidden.to("cpu")).all()
    assert hidden_cos >= MIN_COSINE, f"block hidden cosine {hidden_cos:.5f}"
    assert base_cos >= MIN_COSINE, f"base logit cosine {base_cos:.5f}"
