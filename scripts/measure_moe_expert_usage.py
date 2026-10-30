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


def truncate_routes(records: Sequence[dict], top_k: int) -> list[dict]:
    """Keep the first ``top_k`` experts per token. The router's ``torch.topk``
    sorts by score, so this is what a top-``top_k`` router picks (to first order:
    later layers saw hidden states from the full top-k mixture)."""
    width = records[0]["routes"].shape[-1] if records else top_k
    if not 1 <= top_k <= width:
        raise ValueError(f"top_k must be in [1, {width}], got {top_k}")
    return [{**r, "routes": r["routes"][..., :top_k]} for r in records]


def decode_ms_for_top_k(
    decode_ms: float, model: "CostModel", trace_top_k: int, top_k: int
) -> float:
    """Decode step cost if each token loaded ``top_k`` instead of ``trace_top_k``."""
    return decode_ms - model.per_expert_ms * (trace_top_k - top_k)


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
# Union cap (SPYRE_MOE_UNION_CAP, PR #721): what a fixed C-slot verify keeps
# ---------------------------------------------------------------------------

DEFAULT_CAPS = (8, 12, 16, 24, 32, 40, 48)
P99 = 0.99


@dataclass(frozen=True)
class CapWindowStats:
    """One k-token window through a C-slot union, per layer ``[L]`` / token ``[L, k]``."""

    union: object  # distinct experts the window routes to
    kept: object  # of those, how many got a slot
    dropped_mass: object  # each token's renormalized top-k weight that was dropped
    kept_per_token: object  # each token's routed experts that kept a slot


@dataclass(frozen=True)
class CapRow:
    tokens: int
    cap: int
    windows: int
    mean_union: float
    mean_kept: float
    mean_dropped_experts: float
    kept_fraction: float  # mean over (window, layer) of kept / union
    dropped_mass_mean: float  # over (window, layer, token)
    dropped_mass_p99: float
    dropped_mass_max: float
    fully_kept_fraction: float  # tokens whose whole top-k kept a slot
    top1_only_fraction: float  # tokens left with only one expert
    verify_ms: float
    verify_over_decode: float


def _dense_top_k(probs, top_k: int):
    """Router softmax ``[..., E]`` -> renormalized top-k weights, zeros elsewhere."""
    import torch

    values, index = probs.topk(top_k, dim=-1)
    kept = torch.zeros_like(probs).scatter(-1, index, values)
    return kept / kept.sum(dim=-1, keepdim=True)


def union_slot_mask(probs, top_k: int, cap: int):
    """#721's slot choice for ``probs`` ``[L, T, E]``: bool ``[L, E]``.

    Same rule as ``hf_common.moe_union_pack``: every token's top-1 expert first
    (ties count as top-1, as ``>= amax`` does there), then the experts with the
    most renormalized route weight summed over the tokens.
    """
    import torch

    tokens, experts = probs.shape[-2:]
    dense = _dense_top_k(probs, top_k)
    top1 = (probs >= probs.amax(dim=-1, keepdim=True)).to(dense.dtype)
    score = top1.amax(dim=-2) + dense.sum(dim=-2) / (tokens + 1)
    slots = score.topk(min(cap, experts), dim=-1).indices
    return torch.zeros(score.shape, dtype=torch.bool).scatter(-1, slots, True)


def cap_window_stats(probs, top_k: int, cap: int) -> CapWindowStats:
    """What a ``cap``-slot union keeps of one window ``probs`` ``[L, T, E]``."""
    tokens = probs.shape[-2]
    if tokens > cap:
        raise ValueError(f"union form needs tokens ({tokens}) <= cap ({cap})")
    dense = _dense_top_k(probs.float(), top_k)
    slots = union_slot_mask(probs.float(), top_k, cap)
    routed = dense > 0
    union = routed.any(dim=-2)
    kept_routes = routed & slots[:, None, :]
    return CapWindowStats(
        union=union.sum(dim=-1),
        kept=(union & slots).sum(dim=-1),
        dropped_mass=(1.0 - (dense * slots[:, None, :]).sum(dim=-1)).clamp_min(0.0),
        kept_per_token=kept_routes.sum(dim=-1),
    )


def _generated_windows(records: Sequence[dict], k: int):
    """Every k-token window of generated tokens: ``probs`` ``[L, k, E]``."""
    for record in records:
        probs, first = record["probs"], int(record["prompt_len"])
        for start in range(first, probs.shape[1] - k + 1):
            yield probs[:, start : start + k]


def summarize_caps(
    records: Sequence[dict],
    windows: Sequence[int],
    caps: Sequence[int],
    *,
    top_k: int,
    model: CostModel,
    head_ms: float,
    decode_ms: float,
) -> list[CapRow]:
    """One row per (k, cap); caps below k are skipped (the union form refuses them).

    Verify cost uses ``cap``, not the kept count: the union form reads every slot.
    """
    import torch

    rows = []
    for k in windows:
        stats_by_cap: dict[int, list[CapWindowStats]] = {c: [] for c in caps if c >= k}
        for window in _generated_windows(records, k):
            for cap, stats in stats_by_cap.items():
                stats.append(cap_window_stats(window, top_k, cap))
        for cap, stats in stats_by_cap.items():
            if not stats:
                raise ValueError(f"no {k}-token windows; continuations are too short")
            union = torch.stack([s.union for s in stats]).float()
            kept = torch.stack([s.kept for s in stats]).float()
            mass = torch.stack([s.dropped_mass for s in stats]).flatten()
            per_token = torch.stack([s.kept_per_token for s in stats]).flatten()
            verify_ms = model.backbone_ms(cap) + head_ms
            rows.append(
                CapRow(
                    tokens=k,
                    cap=cap,
                    windows=len(stats),
                    mean_union=float(union.mean()),
                    mean_kept=float(kept.mean()),
                    mean_dropped_experts=float((union - kept).mean()),
                    kept_fraction=float((kept / union).mean()),
                    dropped_mass_mean=float(mass.mean()),
                    dropped_mass_p99=float(mass.quantile(P99)),
                    dropped_mass_max=float(mass.max()),
                    fully_kept_fraction=float((per_token == top_k).float().mean()),
                    top1_only_fraction=float((per_token == 1).float().mean()),
                    verify_ms=verify_ms,
                    verify_over_decode=verify_ms / decode_ms,
                )
            )
    return rows


def format_cap_table(rows: Sequence[CapRow], top_k: int) -> str:
    header = (
        f"{'k':>3} {'C':>3} {'windows':>8} {'union':>6} {'kept':>5} {'drop':>5} "
        f"{'kept %':>7} | {'mass mean':>9} {'p99':>6} {'max':>6} | "
        f"{'all-k %':>7} {'top1 %':>6} | {'verify ms':>9} {'v/d':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.tokens:>3} {r.cap:>3} {r.windows:>8} {r.mean_union:>6.1f} "
            f"{r.mean_kept:>5.1f} {r.mean_dropped_experts:>5.1f} "
            f"{100 * r.kept_fraction:>6.1f}% | {r.dropped_mass_mean:>9.4f} "
            f"{r.dropped_mass_p99:>6.3f} {r.dropped_mass_max:>6.3f} | "
            f"{100 * r.fully_kept_fraction:>6.1f}% {100 * r.top1_only_fraction:>5.1f}% | "
            f"{r.verify_ms:>9.1f} {r.verify_over_decode:>5.2f}"
        )
    lines.append(
        "union / kept / drop = distinct experts per layer the window routes to / that "
        "got one of the C slots / that did not. mass = a token's renormalized top-"
        f"{top_k} weight on dropped experts (#721 renormalizes it away). all-k % = "
        f"tokens with their whole top-{top_k} kept; top1 % = tokens left with one "
        "expert. verify ms = cost model at C experts (every slot is read)."
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
    captured_probs: list = []

    def hook(_module, _inputs, output):
        captured.append(output[2].detach().to(torch.int16).cpu())  # top_k_index
        if args.save_probs:
            captured_probs.append(output[0].detach().float().cpu())  # softmax [T, E]

    handles = [r.register_forward_hook(hook) for r in routers]
    records = []
    try:
        for i, record in enumerate(dump["records"]):
            start = time.perf_counter()
            captured.clear()
            captured_probs.clear()
            with torch.no_grad():
                model(record["input_ids"][None])
            if len(captured) != len(routers):
                raise RuntimeError(
                    f"{len(captured)} router calls, {len(routers)} routers"
                )
            entry = {
                "prompt_len": int(record["prompt_len"]),
                "routes": torch.stack(captured),
            }
            if args.save_probs:
                entry["probs"] = torch.stack(captured_probs)  # [L, T, E]
            records.append(entry)
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
    num_experts, trace_top_k = data["num_experts"], data["top_k"]
    # The spike measured decode with the trace's (trained) top_k.
    model = CostModel.from_points(
        decode_backbone_ms=args.decode_backbone_ms,
        decode_experts=trace_top_k,
        all_backbone_ms=args.all_backbone_ms,
        all_experts=num_experts,
    )
    top_k = args.top_k or trace_top_k
    records = truncate_routes(data["records"], top_k)
    decode_ms = decode_ms_for_top_k(args.decode_ms, model, trace_top_k, top_k)
    if top_k != trace_top_k:
        _log(f"top_k {trace_top_k} -> {top_k}: decode {decode_ms:.1f} ms (model)")
    rows = summarize_windows(
        records,
        args.windows,
        num_experts=num_experts,
        top_k=top_k,
        model=model,
        head_ms=args.verify_head_ms,
        decode_ms=decode_ms,
    )
    print(format_table(rows, num_experts))
    print(
        f"\ncost model: backbone = {model.fixed_ms:.1f} + {model.per_expert_ms:.2f} ms"
        f" x E; verify = backbone + {args.verify_head_ms:.1f} ms head; "
        f"decode = {decode_ms:.1f} ms, top_k={top_k} (best case: used-experts-only kernel)"
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(ratios_json(rows), indent=2))
        _log(f"wrote {args.out_json}")


def run_cap_report(args: argparse.Namespace) -> None:
    import torch

    data = torch.load(args.routes, weights_only=True)
    records = data["records"]
    if not records or any("probs" not in r for r in records):
        raise SystemExit(
            f"{args.routes} has no router probabilities; re-run collect --save-probs"
        )
    num_experts, top_k = data["num_experts"], data["top_k"]
    model = CostModel.from_points(
        decode_backbone_ms=args.decode_backbone_ms,
        decode_experts=top_k,
        all_backbone_ms=args.all_backbone_ms,
        all_experts=num_experts,
    )
    rows = summarize_caps(
        records,
        args.windows,
        args.caps,
        top_k=top_k,
        model=model,
        head_ms=args.verify_head_ms,
        decode_ms=args.decode_ms,
    )
    print(format_cap_table(rows, top_k))
    print(
        f"\ncost model: backbone = {model.fixed_ms:.1f} + {model.per_expert_ms:.2f} ms"
        f" x C; verify = backbone + {args.verify_head_ms:.1f} ms head; "
        f"decode = {args.decode_ms:.1f} ms"
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps([r.__dict__ for r in rows], indent=2))
        _log(f"wrote {args.out_json}")


def _add_cost_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decode-backbone-ms", type=float, default=DEFAULT_DECODE_BACKBONE_MS
    )
    parser.add_argument(
        "--all-backbone-ms", type=float, default=DEFAULT_ALL_BACKBONE_MS
    )
    parser.add_argument("--verify-head-ms", type=float, default=DEFAULT_VERIFY_HEAD_MS)
    parser.add_argument("--decode-ms", type=float, default=DEFAULT_DECODE_MS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    collect = sub.add_parser("collect", help="record router top-k on CPU")
    collect.add_argument("--taps", required=True, help="measure_xpress_acceptance dump")
    collect.add_argument("--target-path", help="default: the dump's target")
    collect.add_argument("--out", required=True)
    collect.add_argument(
        "--save-probs",
        action="store_true",
        help="also store each router's full softmax (needed by cap-report)",
    )

    report = sub.add_parser("report", help="distinct experts + verify cost per k")
    report.add_argument("--routes", required=True)
    report.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    _add_cost_args(report)
    report.add_argument(
        "--top-k",
        type=int,
        help="re-route with fewer experts per token (default: the trace's top_k)",
    )
    report.add_argument("--out-json")

    cap = sub.add_parser(
        "cap-report", help="what SPYRE_MOE_UNION_CAP=C keeps of each k-token verify"
    )
    cap.add_argument("--routes", required=True, help="collect --save-probs output")
    cap.add_argument("--windows", type=int, nargs="+", default=[4, 6, 8, 16])
    cap.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    _add_cost_args(cap)
    cap.add_argument("--out-json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stages = {
        "collect": run_collect,
        "report": run_report,
        "cap-report": run_cap_report,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
