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

"""How many distinct MoE experts does a k-token verify load? (Gemma4, #610)

The verify-cost spike showed the 16-token XPress verify on Gemma4 26B-A4B is
bound by loading expert weights, not by rows: 64 and 128 rows cost the same,
because today's multi-token path runs all 128 experts. A kernel that loads only
the experts the verify tokens route to would cost roughly in proportion to the
number of *distinct* experts per layer, E. This script measures E on real
Gemma4 routing and turns it into an estimated verify/decode ratio per verify
size k:

``collect``  Stock HF target on CPU over the sequences ``measure_xpress_
             acceptance.py dump`` saved; a forward hook on every
             ``Gemma4TextRouter`` records each token's top-k expert ids
             (``[layers, tokens, top_k]`` per prompt).
``report``   For every window of k consecutive generated tokens, E = distinct
             experts per layer (mean over layers, since verify cost sums over
             layers); mean / p90 over windows, next to the random-routing
             expectation. Cost model, from two spike points (decode loads
             ``top_k`` experts, the all-experts verify loads all of them)::

                 backbone(E) = fixed + per_expert * E
                 verify/decode(k) = (backbone(E(k)) + verify LM head) / decode

             This is a best case: it assumes a used-experts-only kernel pays
             the same per expert as today's all-experts path.

Usage::

    python scripts/measure_moe_expert_usage.py collect \\
        --taps ~/xpress-al/taps.pt --out ~/xpress-al/routes.pt
    python scripts/measure_moe_expert_usage.py report \\
        --routes ~/xpress-al/routes.pt --out-json ~/xpress-al/verify_cost.json
    # then: measure_xpress_acceptance.py replay ... --verify-tokens 4 6 8 16 \\
    #       --verify-cost-json ~/xpress-al/verify_cost.json
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# x86 pod, current main (2026-10-22): scripts/spike_verify_cost_spyre.py
DEFAULT_DECODE_BACKBONE_MS = 64.06  # loads top_k experts per layer
DEFAULT_ALL_BACKBONE_MS = 953.68  # 64-row verify, loads every expert
DEFAULT_DECODE_MS = 76.76  # backbone + 1-row LM head
DEFAULT_VERIFY_HEAD_MS = 14.18  # LM head over the verify rows
DEFAULT_WINDOWS = (1, 4, 6, 8, 16)
P90 = 0.9


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU: tests/cpu/test_measure_moe_expert_usage.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Backbone ms as a linear function of distinct experts loaded per layer."""

    fixed_ms: float
    per_expert_ms: float

    @classmethod
    def from_points(
        cls,
        *,
        decode_backbone_ms: float,
        decode_experts: int,
        all_backbone_ms: float,
        all_experts: int,
    ) -> "CostModel":
        if all_experts == decode_experts:
            raise ValueError("the two spike points need distinct expert counts")
        per_expert = (all_backbone_ms - decode_backbone_ms) / (
            all_experts - decode_experts
        )
        return cls(decode_backbone_ms - per_expert * decode_experts, per_expert)

    def backbone_ms(self, experts: float) -> float:
        return self.fixed_ms + self.per_expert_ms * experts


@dataclass(frozen=True)
class WindowRow:
    """E and the estimated verify cost for verifies of ``tokens`` tokens."""

    tokens: int
    windows: int
    mean_experts: float
    p90_experts: float
    random_experts: float
    verify_ms: float
    verify_over_decode: float
    # Static shapes: a correct kernel loads the busiest window's experts.
    per_layer_slots: float  # mean over layers of each layer's max E
    uniform_slots: int  # one slot count for every layer: the overall max E
    static_verify_over_decode: float  # with per_layer_slots
    uniform_verify_over_decode: float  # with uniform_slots


def distinct_per_layer(routes, start: int, k: int):
    """Distinct experts per layer among tokens ``[start, start + k)``.

    ``routes``: ``[layers, tokens, top_k]`` expert ids. Returns ``[layers]``.
    """
    import torch

    layers, tokens, _ = routes.shape
    if start < 0 or k < 1 or start + k > tokens:
        raise ValueError(f"window [{start}, {start + k}) outside {tokens} tokens")
    ids = routes[:, start : start + k].reshape(layers, -1).long()
    used = torch.zeros(layers, int(routes.max()) + 1, dtype=torch.bool)
    return used.scatter(1, ids, True).sum(dim=1)


def window_counts(records: Sequence[dict], k: int):
    """E per layer for every k-token window of generated tokens (all prompts):
    ``[windows, layers]``."""
    import torch

    rows = []
    layers = 0
    for record in records:
        routes, first = record["routes"], int(record["prompt_len"])
        layers = routes.shape[0]
        for start in range(first, routes.shape[1] - k + 1):
            rows.append(distinct_per_layer(routes, start, k))
    if not rows:
        return torch.zeros(0, layers, dtype=torch.long)
    return torch.stack(rows)


def window_means(records: Sequence[dict], k: int) -> list[float]:
    """Layer-mean E for every k-token window of generated tokens (all prompts)."""
    return window_counts(records, k).float().mean(dim=1).tolist()


def random_distinct(num_experts: int, top_k: int, k: int) -> float:
    """Expected distinct experts if each token picked top_k uniformly at random."""
    return num_experts * (1 - (1 - top_k / num_experts) ** k)


def verify_over_decode(
    model: CostModel, experts: float, *, head_ms: float, decode_ms: float
) -> float:
    return (model.backbone_ms(experts) + head_ms) / decode_ms


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (``q`` in [0, 1])."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_windows(
    records: Sequence[dict],
    ks: Sequence[int],
    *,
    num_experts: int,
    top_k: int,
    model: CostModel,
    head_ms: float,
    decode_ms: float,
) -> list[WindowRow]:
    def ratio(experts: float) -> float:
        return verify_over_decode(model, experts, head_ms=head_ms, decode_ms=decode_ms)

    rows = []
    for k in ks:
        counts = window_counts(records, k)
        if counts.shape[0] == 0:
            raise ValueError(f"no {k}-token windows; continuations are too short")
        means = counts.float().mean(dim=1).tolist()
        mean = statistics.fmean(means)
        per_layer = float(counts.max(dim=0).values.float().mean())
        uniform = int(counts.max())
        rows.append(
            WindowRow(
                tokens=k,
                windows=len(means),
                mean_experts=mean,
                p90_experts=percentile(means, P90),
                random_experts=random_distinct(num_experts, top_k, k),
                verify_ms=model.backbone_ms(mean) + head_ms,
                verify_over_decode=ratio(mean),
                per_layer_slots=per_layer,
                uniform_slots=uniform,
                static_verify_over_decode=ratio(per_layer),
                uniform_verify_over_decode=ratio(uniform),
            )
        )
    return rows


def ratios_json(rows: Sequence[WindowRow]) -> dict:
    """The per-k verify/decode map ``measure_xpress_acceptance.py`` reads."""
    return {
        "verify_over_decode": {str(r.tokens): r.verify_over_decode for r in rows},
        "verify_over_decode_static": {
            str(r.tokens): r.static_verify_over_decode for r in rows
        },
        "verify_over_decode_uniform": {
            str(r.tokens): r.uniform_verify_over_decode for r in rows
        },
        "mean_experts": {str(r.tokens): r.mean_experts for r in rows},
    }


def format_table(rows: Sequence[WindowRow], num_experts: int) -> str:
    header = (
        f"{'k':>3} {'windows':>8} {'E mean':>7} {'E p90':>6} {'E random':>9} "
        f"{'verify ms':>10} {'v/d':>6} | {'S/layer':>7} {'v/d':>6} | "
        f"{'S max':>5} {'v/d':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.tokens:>3} {r.windows:>8} {r.mean_experts:>7.1f} "
            f"{r.p90_experts:>6.1f} {r.random_experts:>9.1f} "
            f"{r.verify_ms:>10.1f} {r.verify_over_decode:>6.2f} | "
            f"{r.per_layer_slots:>7.1f} {r.static_verify_over_decode:>6.2f} | "
            f"{r.uniform_slots:>5} {r.uniform_verify_over_decode:>6.2f}"
        )
    lines.append(
        f"E = distinct experts per layer (of {num_experts}); mean = ideal dynamic "
        "kernel. Static shapes must cover the busiest window seen: S/layer = each "
        "layer sized to its own max; S max = one size for all layers. Max over this "
        "sample only: a real kernel still needs an overflow fallback."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[moe-expert-usage] {message}", file=sys.stderr, flush=True)


def _find_routers(model) -> list:
    routers = [m for m in model.modules() if type(m).__name__.endswith("TextRouter")]
    if not routers:
        raise ValueError(f"no *TextRouter modules in {type(model).__name__}")
    return routers


def run_collect(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    dump = torch.load(args.taps, weights_only=True)
    target = args.target_path or dump["target_path"]
    _log(f"loading {target} (bf16, CPU)")
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16).eval()
    text_config = model.config.get_text_config()
    routers = _find_routers(model)
    captured: list = []

    def hook(_module, _inputs, output):
        captured.append(output[2].detach().to(torch.int16).cpu())  # top_k_index

    handles = [r.register_forward_hook(hook) for r in routers]
    records = []
    try:
        for i, record in enumerate(dump["records"]):
            start = time.perf_counter()
            captured.clear()
            with torch.no_grad():
                model(record["input_ids"][None])
            if len(captured) != len(routers):
                raise RuntimeError(
                    f"{len(captured)} router calls, {len(routers)} routers"
                )
            records.append(
                {
                    "prompt_len": int(record["prompt_len"]),
                    "routes": torch.stack(captured),
                }
            )
            _log(
                f"prompt {i}: {record['input_ids'].numel()} tokens, "
                f"{time.perf_counter() - start:.0f} s"
            )
    finally:
        for handle in handles:
            handle.remove()
    torch.save(
        {
            "target_path": target,
            "num_experts": int(text_config.num_experts),
            "top_k": int(text_config.top_k_experts),
            "records": records,
        },
        args.out,
    )
    _log(
        f"wrote routes for {len(records)} prompts x {len(routers)} layers to {args.out}"
    )


def run_report(args: argparse.Namespace) -> None:
    import torch

    data = torch.load(args.routes, weights_only=True)
    num_experts, top_k = data["num_experts"], data["top_k"]
    model = CostModel.from_points(
        decode_backbone_ms=args.decode_backbone_ms,
        decode_experts=top_k,
        all_backbone_ms=args.all_backbone_ms,
        all_experts=num_experts,
    )
    rows = summarize_windows(
        data["records"],
        args.windows,
        num_experts=num_experts,
        top_k=top_k,
        model=model,
        head_ms=args.verify_head_ms,
        decode_ms=args.decode_ms,
    )
    print(format_table(rows, num_experts))
    print(
        f"\ncost model: backbone = {model.fixed_ms:.1f} + {model.per_expert_ms:.2f} ms"
        f" x E; verify = backbone + {args.verify_head_ms:.1f} ms head; "
        f"decode = {args.decode_ms:.1f} ms (best case: used-experts-only kernel)"
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(ratios_json(rows), indent=2))
        _log(f"wrote {args.out_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    collect = sub.add_parser("collect", help="record router top-k on CPU")
    collect.add_argument("--taps", required=True, help="measure_xpress_acceptance dump")
    collect.add_argument("--target-path", help="default: the dump's target")
    collect.add_argument("--out", required=True)

    report = sub.add_parser("report", help="distinct experts + verify cost per k")
    report.add_argument("--routes", required=True)
    report.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    report.add_argument(
        "--decode-backbone-ms", type=float, default=DEFAULT_DECODE_BACKBONE_MS
    )
    report.add_argument(
        "--all-backbone-ms", type=float, default=DEFAULT_ALL_BACKBONE_MS
    )
    report.add_argument("--verify-head-ms", type=float, default=DEFAULT_VERIFY_HEAD_MS)
    report.add_argument("--decode-ms", type=float, default=DEFAULT_DECODE_MS)
    report.add_argument("--out-json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "collect":
        run_collect(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()
