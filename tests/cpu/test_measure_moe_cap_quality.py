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

"""CPU tests for ``scripts/measure_moe_cap_quality.py`` (what a union cap costs
the target's next-token predictions)."""

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPTS = Path(__file__).parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cq():
    return _load("measure_moe_cap_quality")


@pytest.fixture(scope="module")
def usage():
    return _load("measure_moe_expert_usage")


def _probs(tokens, experts, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(tokens, experts, generator=g) * 2, dim=-1)


# ---------------------------------------------------------------------------
# Keep masks
# ---------------------------------------------------------------------------


def test_prompt_rows_keep_every_expert(cq):
    keep = cq.block_keep_mask(_probs(10, 16), prompt_len=4, block=3, cap=4, top_k=2)
    assert keep[:4].all()
    assert not keep[4:].all()


def test_each_block_shares_its_union_slots(cq, usage):
    probs = _probs(10, 16)
    keep = cq.block_keep_mask(probs, prompt_len=4, block=3, cap=4, top_k=2)
    # Generated rows 4..9 -> blocks [4, 7), [7, 10).
    for lo, hi in [(4, 7), (7, 10)]:
        slots = usage.union_slot_mask(probs[None, lo:hi], 2, 4)[0]
        assert all(torch.equal(keep[r], slots) for r in range(lo, hi))
        assert int(slots.sum()) == 4


def test_short_last_block_is_capped_on_its_own(cq, usage):
    probs = _probs(9, 16)
    keep = cq.block_keep_mask(probs, prompt_len=4, block=3, cap=4, top_k=2)
    slots = usage.union_slot_mask(probs[None, 7:9], 2, 4)[0]
    assert torch.equal(keep[7], slots) and torch.equal(keep[8], slots)


def test_cap_below_block_is_refused(cq):
    with pytest.raises(ValueError, match="cap"):
        cq.block_keep_mask(_probs(10, 16), prompt_len=4, block=4, cap=3, top_k=2)


# ---------------------------------------------------------------------------
# Router weights
# ---------------------------------------------------------------------------


def _hf_weights(probs, top_k, scale):
    """Stock ``Gemma4TextRouter`` weights for router softmax ``probs``."""
    w, idx = probs.topk(top_k, dim=-1)
    return w / w.sum(-1, keepdim=True) * scale[idx], idx


def test_keeping_everything_reproduces_the_stock_router(cq):
    probs, scale = _probs(6, 16), torch.rand(16) + 0.5
    stock, idx = _hf_weights(probs, 4, scale)
    keep = torch.ones(6, 16, dtype=torch.bool)
    assert torch.allclose(cq.capped_weights(probs, idx, scale, keep), stock)


def test_dropped_weight_is_renormalized_away(cq):
    probs, scale = _probs(1, 8), torch.rand(8) + 0.5
    stock, idx = _hf_weights(probs, 4, scale)
    keep = torch.ones(1, 8, dtype=torch.bool)
    keep[0, idx[0, 1]] = False  # drop the token's 2nd expert
    out = cq.capped_weights(probs, idx, scale, keep)
    assert out[0, 1] == 0
    norm = (stock / scale[idx])[0]
    kept = [0, 2, 3]
    expected = norm[kept] / norm[kept].sum() * scale[idx[0, kept]]
    assert torch.allclose(out[0, kept], expected)


def test_fully_dropped_row_gives_zeros_not_nan(cq):
    probs, scale = _probs(1, 8), torch.ones(8)
    _, idx = probs.topk(2, dim=-1)
    out = cq.capped_weights(probs, idx, scale, torch.zeros(1, 8, dtype=torch.bool))
    assert torch.equal(out, torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_identical_logits_are_lossless(cq):
    logits = torch.randn(5, 11)
    m = cq.prediction_metrics(logits, logits.clone(), logits.argmax(-1))
    assert m == pytest.approx({"top1_agree": 1.0, "kl": 0.0, "dnll": 0.0}, abs=1e-6)


def test_metrics_on_a_known_flip(cq):
    ref = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    cap = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # row 1 flips
    m = cq.prediction_metrics(ref, cap, torch.tensor([0, 0]))
    assert m["top1_agree"] == pytest.approx(0.5)
    assert m["kl"] > 0 and m["dnll"] > 0


# ---------------------------------------------------------------------------
# Hooks on a tiny random Gemma4 MoE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        enable_moe_block=True,
        num_experts=16,
        top_k_experts=4,
        moe_intermediate_size=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).eval()
    for router in cq_routers(model):
        router.per_expert_scale.data = torch.rand(16) + 0.5
    return model


def cq_routers(model):
    return [m for m in model.modules() if type(m).__name__.endswith("TextRouter")]


@torch.no_grad()
def test_cap_covering_all_experts_changes_nothing(cq, tiny):
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(1))
    ref = tiny(ids).logits
    with cq.union_cap(tiny, prompt_len=8, block=4, cap=16):
        capped = tiny(ids).logits
    assert torch.allclose(ref, capped, atol=1e-5)


@torch.no_grad()
def test_small_cap_changes_only_generated_positions(cq, tiny):
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(1))
    ref = tiny(ids).logits
    with cq.union_cap(tiny, prompt_len=8, block=4, cap=4) as stats:
        capped = tiny(ids).logits
    # Causal: prompt positions never see a capped row.
    assert torch.allclose(ref[:, :8], capped[:, :8], atol=1e-5)
    assert not torch.allclose(ref[:, 8:], capped[:, 8:], atol=1e-3)
    assert 0 < stats.mean_dropped < 1
    # Hooks are gone after the context.
    assert torch.allclose(tiny(ids).logits, ref, atol=1e-6)
