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

"""Spike (issue #610, PR-2): is a 16-token target verify cheap enough for XPress?

XPress proposes 15 tokens per block; the Gemma 4 26B-A4B target then verifies
the anchor + 15 drafts in one forward. That only pays off if a ``V``-token
forward costs not much more than a 1-token decode step.

The verify step is measured as a chunked-prefill chunk: with
``prefill_chunk_size=R`` every chunk after the first is ``R`` new rows
attending to the KV cache plus themselves (causally) and writing ``R`` cache
slots — the verify computation minus rollback. A multi-row query must be a
whole number of 64-wide sticks (``generate`` enforces it), so a 16-token block
runs as ``R = 64`` rows (16 real + 48 masked) today; ``--block-tokens`` (16)
is what one block can yield. The default prompt length (448) keeps the key
width at 512 for both prefill chunks and decode steps, so the two attend over
the same ``Lk``.

What is timed (device-synchronized, after a full warm-up ``generate``):

- backbone: ``_run_backbone_forward`` (embedding -> 30 blocks -> norm), for
  every decode step (1 token) and every prefill chunk but the first (V tokens);
- LM head: ``run_lm_head`` over 1 row and over V rows.

``--moe-path``:

- ``prefill`` (default, today's code): a multi-token forward evaluates **all**
  128 experts (``moe_prefill_all_experts``);
- ``gather`` (experimental): route multi-token forwards through the decode
  graph, which gathers only each token's top-8 experts
  (``moe_decode_selected_experts``). May fail to compile; that is a finding.

``--row-multiple 16`` (experimental) lowers ``hf_common.BLOCK_SIZE`` after
the model is prepared, so ``generate`` accepts a 16-row chunk: the verify
runs without the 48 padding rows. Weights and caches are already laid out by
then; only the chunk guard and prompt padding see the smaller value.

``--split-timing`` additionally times every block's compiled attention and
FFN (MoE + dense) graph for multi-token forwards, with a device sync around
each (slower overall; use it for the split, not the headline numbers).

Reports the per-token decode cost, the verify cost and their ratio, and the
break-even acceptance length: the tokens a block must yield for
``verify + draft`` to beat that many plain decode steps.

Usage (on the Spyre pod, project root on PYTHONPATH)::

    python scripts/spike_verify_cost_spyre.py
    python scripts/spike_verify_cost_spyre.py --verify-rows 128
    python scripts/spike_verify_cost_spyre.py --moe-path gather
    python scripts/spike_verify_cost_spyre.py --verify-rows 16 --row-multiple 16
    python scripts/spike_verify_cost_spyre.py --split-timing
    # tensor parallel, one process per card:
    SPYRE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 \\
        scripts/spike_verify_cost_spyre.py
"""

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_PROMPT_TOKENS = 448  # prefill Lk and decode Lk both round to 512
DEFAULT_VERIFY_ROWS = 64  # one stick: the smallest multi-row query Spyre takes
DEFAULT_BLOCK_TOKENS = 16  # XPress block: anchor + 15 drafts
DEFAULT_DECODE_STEPS = 32
DEFAULT_DRAFT_MS = 50.4  # XPress propose on Spyre, bf16, topc 512 (PR #677)
DEFAULT_HEAD_REPS = 10
PROMPT_SEED = "The quick brown fox jumps over the lazy dog. "
SDPA_KV_BLOCK = 512  # hf_common._SDPA_MAX_SEQUENCE_TILE_SIZE
STICK = 64  # hf_common.BLOCK_SIZE
ROW_MULTIPLES = (16, 32, STICK)  # --row-multiple choices


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU: tests/cpu/test_spike_verify_cost.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpikeResult:
    """Median costs (ms) and the derived verify/decode comparison."""

    decode_backbone_ms: float
    verify_backbone_ms: float
    decode_head_ms: float
    verify_head_ms: float
    draft_ms: float
    verify_rows: int
    block_tokens: int

    @property
    def decode_ms(self) -> float:
        return self.decode_backbone_ms + self.decode_head_ms

    @property
    def verify_ms(self) -> float:
        return self.verify_backbone_ms + self.verify_head_ms

    @property
    def verify_over_decode(self) -> float:
        return self.verify_ms / self.decode_ms

    @property
    def break_even_tokens(self) -> float:
        """Tokens per block at which ``verify + draft`` equals plain decode."""
        return (self.verify_ms + self.draft_ms) / self.decode_ms

    def speedup_at(self, tokens_per_block: float) -> float:
        """Decode speedup if each block yields ``tokens_per_block`` tokens."""
        return tokens_per_block * self.decode_ms / (self.verify_ms + self.draft_ms)


def median_ms(samples: list[float], *, label: str) -> float:
    if not samples:
        raise RuntimeError(f"no {label} samples were recorded")
    return statistics.median(samples)


def split_backbone_samples(
    records: list[tuple[int, float]], verify_rows: int
) -> tuple[list[float], list[float]]:
    """Split ``(seq_len, ms)`` backbone records into decode and verify samples.

    Decode = every 1-token call. Verify = every ``verify_rows`` call except
    the first, which starts a fresh prefill (empty cache, state reset).
    """
    decode = [ms for seq_len, ms in records if seq_len == 1]
    chunks = [ms for seq_len, ms in records if seq_len == verify_rows]
    return decode, chunks[1:]


def check_geometry(
    prompt_tokens: int,
    verify_rows: int,
    block_tokens: int,
    decode_steps: int,
    row_multiple: int = STICK,
) -> None:
    """Fail fast on settings that would not measure a verify-like chunk."""
    if row_multiple not in ROW_MULTIPLES:
        raise ValueError(f"--row-multiple must be one of {ROW_MULTIPLES}")
    if verify_rows <= 0 or verify_rows % row_multiple:
        raise ValueError(f"--verify-rows must be a positive multiple of {row_multiple}")
    if not 2 <= block_tokens <= verify_rows:
        raise ValueError("--block-tokens must be in [2, --verify-rows]")
    if decode_steps < 1:
        raise ValueError("--decode-steps must be >= 1")
    if prompt_tokens % verify_rows:
        raise ValueError(
            f"--prompt-tokens ({prompt_tokens}) must be a multiple of "
            f"--verify-rows ({verify_rows}) so every chunk is full"
        )
    if prompt_tokens // verify_rows < 2:
        raise ValueError("need at least 2 prefill chunks (the first is discarded)")
    prefill_lk, decode_lk = key_widths(prompt_tokens, decode_steps + 1, row_multiple)
    if prefill_lk != decode_lk:
        print(
            f"  note: prefill Lk {prefill_lk} != decode Lk {decode_lk}; verify and "
            "decode attend over different key widths",
            file=sys.stderr,
        )


def key_widths(
    prompt_tokens: int, max_new_tokens: int, row_multiple: int = STICK
) -> tuple[int, int]:
    """``(prefill Lk, decode Lk)``, mirroring ``hf_common.generate``'s sizing.

    Chunked prefill: ``_sdpa_compatible_kv_length(prompt)``. Decode:
    ``generation_cache_len(prompt, max_new_tokens)``, whose rounding follows
    ``hf_common.BLOCK_SIZE`` (``row_multiple`` under ``--row-multiple``).
    """

    def round_up(n: int, block: int) -> int:
        return -(-n // block) * block

    prefill = round_up(prompt_tokens, SDPA_KV_BLOCK)
    decode = round_up(
        round_up(prompt_tokens, row_multiple) + round_up(max_new_tokens, row_multiple),
        SDPA_KV_BLOCK,
    )
    return prefill, decode


def format_report(result: SpikeResult, *, samples: tuple[int, int]) -> str:
    rows, block = result.verify_rows, result.block_tokens
    lines = [
        f"  samples: decode={samples[0]}, verify chunks={samples[1]}",
        f"  {'':28}{'backbone':>10}{'lm_head':>10}{'total':>10}",
        f"  {'decode, 1 token (ms)':28}{result.decode_backbone_ms:10.2f}"
        f"{result.decode_head_ms:10.2f}{result.decode_ms:10.2f}",
        f"  {f'verify, {rows} rows (ms)':28}{result.verify_backbone_ms:10.2f}"
        f"{result.verify_head_ms:10.2f}{result.verify_ms:10.2f}",
        f"  verify / decode            : {result.verify_over_decode:.2f}x",
        f"  draft (XPress propose)     : {result.draft_ms:.1f} ms",
        f"  break-even tokens/block    : {result.break_even_tokens:.2f} "
        f"(a block yields at most {block})",
        "  speedup at tokens/block    : "
        + ", ".join(f"{t}->{result.speedup_at(t):.2f}x" for t in (3, 4, 5, 6, 8)),
    ]
    return "\n".join(lines)


def summarize_stages(
    records: list[tuple[str, int, float]], forwards: dict[int, int]
) -> dict[tuple[str, int], float]:
    """Mean ms per forward for each ``(stage, rows)``, summed over all blocks.

    ``records`` holds one ``(stage, rows, ms)`` entry per block call;
    ``forwards`` maps rows to the number of backbone forwards of that width.
    """
    totals: dict[tuple[str, int], float] = {}
    for stage, rows, ms in records:
        totals = {**totals, (stage, rows): totals.get((stage, rows), 0.0) + ms}
    return {
        key: total / forwards[key[1]]
        for key, total in totals.items()
        if forwards.get(key[1])
    }


def format_stages(stages: dict[tuple[str, int], float]) -> str:
    lines = ["  per-forward stage time (all blocks, synced per call):"]
    for (stage, rows), ms in sorted(
        stages.items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        lines = [*lines, f"    {stage:>8} rows={rows:<4} {ms:10.2f} ms"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spyre measurement
# ---------------------------------------------------------------------------


def _synchronize() -> None:
    import torch

    torch.spyre.synchronize()


class BackboneTimer:
    """Wraps ``_run_backbone_forward``; records ``(seq_len, ms)`` per call."""

    def __init__(self, fn, sync=_synchronize):
        self._fn = fn
        self._sync = sync
        self.records: list[tuple[int, float]] = []
        self.last_hidden: dict[int, object] = {}

    def __call__(self, model, input_ids, *args, **kwargs):
        self._sync()
        t0 = time.perf_counter()
        hidden = self._fn(model, input_ids, *args, **kwargs)
        self._sync()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        seq_len = int(input_ids.shape[1])
        self.records = [*self.records, (seq_len, elapsed_ms)]
        self.last_hidden = {**self.last_hidden, seq_len: hidden}
        return hidden

    def reset(self) -> None:
        self.records = []
        self.last_hidden = {}


def install_backbone_timer(module, sync=_synchronize) -> BackboneTimer:
    """Time every backbone call of an adapter: prefill and decode.

    ``generate()`` gets ``module._run_backbone_forward`` as its prefill
    backbone; decode runs ``module._run_forward``, which calls the same
    function through a global of its own module (``hf_gemma4._run_backbone_
    forward`` for Gemma4, ``hf_common.standard_gqa_backbone_forward`` for the
    standard GQA adapters). Every name bound to it there is wrapped too.
    """
    original = module._run_backbone_forward
    timer = BackboneTimer(original, sync=sync)
    module._run_backbone_forward = timer
    decode_globals = module._run_forward.__globals__
    for name, value in list(decode_globals.items()):
        if value is original:
            decode_globals[name] = timer
    return timer


MOE_ADAPTER = "hf_adapters.hf_gemma4_moe"


def check_adapter_options(module_name: str, moe_path: str, split_timing: bool) -> None:
    """``--moe-path gather`` and ``--split-timing`` patch the Gemma4 MoE block."""
    if module_name == MOE_ADAPTER:
        return
    if moe_path != "prefill":
        raise ValueError(f"--moe-path {moe_path} needs the {MOE_ADAPTER} adapter")
    if split_timing:
        raise ValueError(f"--split-timing needs the {MOE_ADAPTER} adapter")


def install_gather_moe_path() -> None:
    """Route multi-token MoE forwards through the gathered decode graph."""
    from hf_adapters.hf_gemma4_moe import Gemma4MoEBlock

    def forward(self, hidden_states, *args):
        return self._compiled_decode(hidden_states, *args)

    Gemma4MoEBlock.forward = forward


class StageTimer:
    """Times each block's compiled attention / FFN / decode graph per call."""

    STAGES = (
        ("attn", "_compiled_prefill_attn"),
        ("ffn", "_compiled_prefill_ffn"),
        ("decode", "_compiled_decode"),
    )

    def __init__(self):
        self.records: list[tuple[str, int, float]] = []

    def wrap(self, stage: str, fn):
        def timed(hidden_states, *args, **kwargs):
            _synchronize()
            t0 = time.perf_counter()
            out = fn(hidden_states, *args, **kwargs)
            _synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            rows = int(hidden_states.shape[1])
            self.records = [*self.records, (stage, rows, elapsed_ms)]
            return out

        return timed

    def reset(self) -> None:
        self.records = []


def install_stage_timer() -> StageTimer:
    """Wrap every ``Gemma4MoEBlock``'s compiled graphs as blocks are built."""
    from hf_adapters.hf_gemma4_moe import Gemma4MoEBlock

    timer = StageTimer()
    original_init = Gemma4MoEBlock.__init__

    def timed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for stage, attr in StageTimer.STAGES:
            setattr(self, attr, timer.wrap(stage, getattr(self, attr)))

    Gemma4MoEBlock.__init__ = timed_init
    return timer


def install_row_multiple(row_multiple: int) -> None:
    """Let ``generate`` take sub-stick prefill chunks (call after loading)."""
    from hf_adapters import hf_common

    hf_common.BLOCK_SIZE = row_multiple


def time_lm_head(model, hidden, reps: int) -> float:
    from hf_adapters.hf_common import run_lm_head

    run_lm_head(model, hidden)  # compile / warm this row count
    samples = []
    for _ in range(reps):
        _synchronize()
        t0 = time.perf_counter()
        run_lm_head(model, hidden)
        _synchronize()
        samples = [*samples, (time.perf_counter() - t0) * 1e3]
    return median_ms(samples, label="lm_head")


def build_input_ids(tokenizer, prompt_tokens: int):
    import torch

    seed = tokenizer(PROMPT_SEED, add_special_tokens=False, return_tensors="pt")
    seed_ids = seed["input_ids"]
    repeats = -(-prompt_tokens // seed_ids.shape[1])
    return torch.cat([seed_ids] * repeats, dim=1)[:, :prompt_tokens]


def run_spike(args: argparse.Namespace) -> SpikeResult:
    import torch
    import torch_spyre  # noqa: F401
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.auto_spyre_model import resolve_adapter_module

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    adapter = resolve_adapter_module(args.model_path)
    check_adapter_options(adapter.__name__, args.moe_path, args.split_timing)
    _log(f"adapter: {adapter.__name__}")
    timer = install_backbone_timer(adapter)
    stage_timer = install_stage_timer() if args.split_timing else None
    if args.moe_path == "gather":
        install_gather_moe_path()

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, tp_plan="auto" if world_size > 1 else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    _log(f"  load: {time.time() - t0:.1f}s  (tp={world_size}, dtype={args.dtype})")
    if args.row_multiple != STICK:
        install_row_multiple(args.row_multiple)

    input_ids = build_input_ids(tokenizer, args.prompt_tokens)
    gen_kwargs = dict(
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=args.decode_steps + 1,
        do_sample=False,
        eos_token_id=None,  # always run every decode step
        prefill_chunk_size=args.verify_rows,
    )
    for phase in ("warm-up (compiles)", "measured"):
        timer.reset()
        if stage_timer is not None:
            stage_timer.reset()
        t0 = time.time()
        model.generate(input_ids, **gen_kwargs)
        _log(f"  generate [{phase}]: {time.time() - t0:.1f}s")

    decode, verify = split_backbone_samples(timer.records, args.verify_rows)
    result = SpikeResult(
        decode_backbone_ms=median_ms(decode, label="decode"),
        verify_backbone_ms=median_ms(verify, label="verify"),
        decode_head_ms=time_lm_head(model, timer.last_hidden[1], args.head_reps),
        verify_head_ms=time_lm_head(
            model, timer.last_hidden[args.verify_rows], args.head_reps
        ),
        draft_ms=args.draft_ms,
        verify_rows=args.verify_rows,
        block_tokens=args.block_tokens,
    )
    _log(format_report(result, samples=(len(decode), len(verify))))
    if stage_timer is not None:
        forwards = {}
        for seq_len, _ in timer.records:
            forwards = {**forwards, seq_len: forwards.get(seq_len, 0) + 1}
        _log(format_stages(summarize_stages(stage_timer.records, forwards)))
    return result


def _log(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument(
        "--verify-rows",
        type=int,
        default=DEFAULT_VERIFY_ROWS,
        help="Query rows of the verify forward (multiple of 64).",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=DEFAULT_BLOCK_TOKENS,
        help="Tokens one draft block can yield (anchor + drafts).",
    )
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument(
        "--moe-path",
        choices=("prefill", "gather"),
        default="prefill",
        help="Multi-token MoE path: all experts (today) or gathered top-k.",
    )
    parser.add_argument(
        "--draft-ms",
        type=float,
        default=DEFAULT_DRAFT_MS,
        help="XPress propose cost to include in the break-even estimate.",
    )
    parser.add_argument("--head-reps", type=int, default=DEFAULT_HEAD_REPS)
    parser.add_argument(
        "--row-multiple",
        type=int,
        choices=ROW_MULTIPLES,
        default=STICK,
        help="Experimental: allow prefill chunks that are a multiple of this.",
    )
    parser.add_argument(
        "--split-timing",
        action="store_true",
        help="Also time each block's attention / FFN graph (adds syncs).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        check_geometry(
            args.prompt_tokens,
            args.verify_rows,
            args.block_tokens,
            args.decode_steps,
            args.row_multiple,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _log(
        f"Verify-cost spike: {args.model_path}, prompt={args.prompt_tokens}, "
        f"verify={args.verify_rows} rows for a {args.block_tokens}-token block, "
        f"moe-path={args.moe_path}, row-multiple={args.row_multiple}, "
        f"split-timing={args.split_timing}"
    )
    run_spike(args)


if __name__ == "__main__":
    main()
