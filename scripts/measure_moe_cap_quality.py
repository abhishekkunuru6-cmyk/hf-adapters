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

"""What does ``SPYRE_MOE_UNION_CAP=C`` cost the target's predictions? (Gemma4, #721)

``measure_moe_expert_usage.py cap-report`` counts what a C-slot union drops at
the router. This script measures what that does to the model's output: the
stock HF Gemma4 forward on CPU, with a forward hook on every
``Gemma4TextRouter`` that applies #721's union cap.

- Prompt tokens are uncapped (the cap only serves 2..C-token decode steps).
- Generated tokens are grouped into consecutive blocks of ``block`` tokens,
  each one verify step that was fully accepted. A block's tokens may only use
  its ``cap`` slot experts, chosen with #721's rule (every token's top-1, then
  most routed weight); weight on other experts is dropped and each row
  renormalized; ``per_expert_scale`` is applied as the stock router does.
- Every layer is capped in the same forward, so errors compound the way they
  do on Spyre: a capped layer changes what later layers route on, and capped
  rows are the KV that later tokens attend to.

Compared with the uncapped forward over the same tokens (the dumped greedy
continuations), on every generated position: top-1 agreement, KL(uncapped ||
capped), and the change in NLL of the reference next token.

Approximation: real verify blocks start wherever the previous step stopped
and also contain rejected drafts; here blocks are fixed-stride and all tokens
are accepted ones. Teacher-forced, so it measures per-step damage, not how far
a free-running generation drifts.

Usage (pod CPU; needs ~55 GB host RAM for the bf16 target)::

    python scripts/measure_moe_cap_quality.py --taps ~/xpress-al/taps.pt \\
        --blocks 8 16 --caps 8 12 16 24 32 48 --out-json cap_quality.json
"""

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BLOCKS = (8, 16)
DEFAULT_CAPS = (8, 12, 16, 24, 32, 48)


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# #721's slot rule, already checked against hf_common.moe_union_pack there.
union_slot_mask = _sibling("measure_moe_expert_usage").union_slot_mask


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU: tests/cpu/test_measure_moe_cap_quality.py)
# ---------------------------------------------------------------------------


def block_keep_mask(probs, *, prompt_len: int, block: int, cap: int, top_k: int):
    """Experts each token may use: bool ``[T, E]`` for router softmax ``[T, E]``.

    Prompt rows keep every expert; generated rows share their block's slots.
    """
    import torch

    if block > cap:
        raise ValueError(f"union form needs block ({block}) <= cap ({cap})")
    keep = torch.ones(probs.shape, dtype=torch.bool)
    for lo in range(prompt_len, probs.shape[0], block):
        hi = min(lo + block, probs.shape[0])
        keep[lo:hi] = union_slot_mask(probs[None, lo:hi].float(), top_k, cap)[0]
    return keep


def capped_weights(probs, top_k_index, per_expert_scale, keep):
    """Stock router weights with experts outside ``keep`` dropped, rows renormalized.

    Stock: ``renorm(top-k softmax) * per_expert_scale[index]``. A row whose every
    expert is dropped gets zeros (as #721's union form does), not ``0 / 0``.
    """
    import torch

    weights = probs.gather(-1, top_k_index)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * keep.gather(-1, top_k_index)
    total = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(total > 0, weights / total.clamp_min(1e-12), 0.0)
    return weights * per_expert_scale[top_k_index]


def prediction_metrics(ref_logits, cap_logits, targets) -> dict[str, float]:
    """Capped vs uncapped next-token predictions over ``N`` positions ``[N, V]``."""
    ref = ref_logits.float().log_softmax(dim=-1)
    cap = cap_logits.float().log_softmax(dim=-1)
    rows = targets[:, None]
    return {
        "top1_agree": float((cap.argmax(-1) == ref.argmax(-1)).float().mean()),
        "kl": float((ref.exp() * (ref - cap)).sum(dim=-1).mean()),
        "dnll": float((ref.gather(-1, rows) - cap.gather(-1, rows)).mean()),
    }


@dataclass
class CapStats:
    """Routing weight dropped per (layer, generated token), filled by the hooks."""

    dropped: list = field(default_factory=list)

    @property
    def mean_dropped(self) -> float:
        import torch

        return float(torch.cat(self.dropped).mean()) if self.dropped else 0.0


def _routers(model) -> list:
    routers = [m for m in model.modules() if type(m).__name__.endswith("TextRouter")]
    if not routers:
        raise ValueError(f"no *TextRouter modules in {type(model).__name__}")
    return routers


@contextmanager
def union_cap(model, *, prompt_len: int, block: int, cap: int) -> Iterator[CapStats]:
    """Apply #721's union cap to every router of ``model`` (batch 1) inside the block."""
    stats = CapStats()

    def hook(router, _inputs, output):
        probs, _, index = output
        keep = block_keep_mask(
            probs, prompt_len=prompt_len, block=block, cap=cap, top_k=index.shape[-1]
        )
        weights = capped_weights(probs, index, router.per_expert_scale, keep)
        kept = (probs.gather(-1, index) * keep.gather(-1, index)).sum(-1)
        stats.dropped.append(
            (1 - kept / probs.gather(-1, index).sum(-1))[prompt_len:].detach()
        )
        return probs, weights.to(output[1].dtype), index

    handles = [r.register_forward_hook(hook) for r in _routers(model)]
    try:
        yield stats
    finally:
        for handle in handles:
            handle.remove()


@dataclass(frozen=True)
class QualityRow:
    block: int
    cap: int
    positions: int
    route_dropped: float  # mean routing weight dropped per (layer, token)
    top1_agree: float
    kl: float
    dnll: float


def config_grid(blocks: Sequence[int], caps: Sequence[int]) -> list[tuple[int, int]]:
    """``(block, cap)`` pairs the union form accepts (``cap >= block``)."""
    return [(b, c) for b in blocks for c in caps if c >= b]


def format_table(rows: Sequence[QualityRow]) -> str:
    header = (
        f"{'k':>3} {'C':>3} {'positions':>9} {'route drop':>10} "
        f"{'top-1 agree':>11} {'KL':>8} {'dNLL':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.block:>3} {r.cap:>3} {r.positions:>9} {r.route_dropped:>10.3f} "
            f"{100 * r.top1_agree:>10.2f}% {r.kl:>8.4f} {r.dnll:>8.4f}"
        )
    lines.append(
        "k = verify block, C = union cap. route drop = routing weight dropped per "
        "(layer, generated token). top-1 agree = capped argmax == uncapped argmax. "
        "KL(uncapped || capped) and dNLL (capped - uncapped NLL of the reference "
        "next token) in nats, mean over generated positions."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[moe-cap-quality] {message}", file=sys.stderr, flush=True)


def _aggregate(
    block: int, cap: int, parts: list[tuple[int, dict, float]]
) -> QualityRow:
    total = sum(n for n, _, _ in parts)

    def mean(key: str) -> float:
        return sum(n * m[key] for n, m, _ in parts) / total

    return QualityRow(
        block=block,
        cap=cap,
        positions=total,
        route_dropped=sum(n * d for n, _, d in parts) / total,
        top1_agree=mean("top1_agree"),
        kl=mean("kl"),
        dnll=mean("dnll"),
    )


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    dump = torch.load(args.taps, weights_only=True)
    records = dump["records"][: args.max_prompts]
    target = args.target_path or dump["target_path"]
    _log(f"{len(records)} records from {args.taps}; loading {target} (bf16, CPU)")
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16).eval()
    grid = config_grid(args.blocks, args.caps)
    parts: dict[tuple[int, int], list] = {cfg: [] for cfg in grid}
    with torch.no_grad():
        for i, record in enumerate(records):
            ids = record["input_ids"][None]
            p = int(record["prompt_len"])
            targets = ids[0, p + 1 :]
            ref = model(ids).logits[0, p:-1]
            for block, cap in grid:
                start = time.perf_counter()
                with union_cap(model, prompt_len=p, block=block, cap=cap) as stats:
                    logits = model(ids).logits[0, p:-1]
                metrics = prediction_metrics(ref, logits, targets)
                parts[(block, cap)].append((len(targets), metrics, stats.mean_dropped))
                _log(
                    f"prompt {i} k={block} C={cap}: top-1 "
                    f"{100 * metrics['top1_agree']:.1f}%, KL {metrics['kl']:.4f} "
                    f"({time.perf_counter() - start:.0f} s)"
                )
    rows = [_aggregate(b, c, parts[(b, c)]) for b, c in grid]
    print(format_table(rows))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps([r.__dict__ for r in rows], indent=2))
        _log(f"wrote {args.out_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--taps", required=True, help="measure_xpress_acceptance dump")
    parser.add_argument("--target-path", help="default: the dump's target")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--blocks", type=int, nargs="+", default=list(DEFAULT_BLOCKS))
    parser.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    parser.add_argument("--out-json")
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
