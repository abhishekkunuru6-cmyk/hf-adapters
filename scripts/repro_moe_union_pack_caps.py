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

"""Standalone repro: PR #721's union-cap slot region fails to compile for some caps.

The Gemma4 26B-A4B tree verify with ``SPYRE_MOE_UNION_CAP=24`` (and 48) dies in
``dbo-opt`` on the union-pack kernel
(``…amax_clamp_min_clone_div_eq_expand_ge_keep_by_index_mm_mul_slice_sum_topk…``)::

    sbf-ddc: DtException: Incorrect chunk size in bytes.
        (deeptools dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1902)

while caps 8, 12, 16, 32 and 64 compile. This script compiles that region,
``hf_common.moe_union_pack`` (``dynamic=False``, ``fullgraph=True``, as the tree
layer runs it), with the real Gemma4 shapes (hidden 2816, 128 experts, top-8,
stick 64), one cap at a time. No checkpoint, no model load. Where the router
softmax comes from (``--probs``):

- ``router`` (default): produced on the device by ``hf_gemma4_moe._router_probs``
  in its own compiled region, then fed to the pack region, the way
  ``_tree_part_a`` -> ``_tree_part_p`` hand it over (device layout included);
- ``host``: a random softmax built on the host and copied to the device.

Each case is checked against the same chain on CPU in fp32: same slot experts
and routes within 0.02. Slot sets that differ only between experts whose fp32
scores sit within 0.5% of the cut-off are reported as ``tie`` (fp16 rounding at
the last slot), not as a failure.

Needs PR #721 (``hf_common.moe_union_pack``) on PYTHONPATH.

Usage (Spyre pod, torch-spyre env)::

    PYTHONPATH=~/hf-adapters-721 python scripts/repro_moe_union_pack_caps.py
    PYTHONPATH=~/hf-adapters-721 python scripts/repro_moe_union_pack_caps.py \\
        --tokens 16 --caps 24 --probs router host -v

``--device cpu`` runs the region eagerly (sanity check of the script itself).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass

import torch

# Gemma4 26B-A4B router.
HIDDEN = 2816
EXPERTS = 128
TOP_K = 8
STICK = 64
EPS = 1e-6
DEFAULT_TOKENS = 16  # #721's tree width
DEFAULT_CAPS = (16, 20, 24, 28, 32, 40, 48)
PROBS_SOURCES = ("router", "host")
ROUTE_TOLERANCE = 0.02  # as #721's tests/spyre/test_moe_union_spyre.py
TIE_RTOL = 5e-3  # slot scores this close to the cut-off may swap under fp16
PROBS_TOLERANCE = 2e-3  # fp16 router softmax vs fp32


@dataclass(frozen=True)
class Result:
    probs: str
    tokens: int
    cap: int
    status: str  # "ok", "tie" or "FAIL"
    seconds: float
    max_route_error: float = float("nan")
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


def _pack():
    from hf_adapters import hf_common

    if not hasattr(hf_common, "moe_union_pack"):
        raise SystemExit("needs PR #721's hf_common.moe_union_pack on PYTHONPATH")
    return hf_common.moe_union_pack


def router_probs(tokens: int, seed: int) -> torch.Tensor:
    """A random fp16 router softmax ``[tokens, EXPERTS]`` (the ``host`` source)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(tokens, EXPERTS, generator=g) * 2).softmax(-1).half()


def router_inputs(tokens: int, seed: int) -> dict:
    """Random fp16 router input and weights with Gemma4 shapes (the ``router`` source)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(tokens, HIDDEN, generator=g).half(),
        "weight": (torch.randn(EXPERTS, HIDDEN, generator=g) * 4 * HIDDEN**-0.5).half(),
        "scale": torch.ones(HIDDEN, dtype=torch.float16),
    }


def router_region(x, weight, scale):
    from hf_adapters.hf_gemma4_moe import _router_probs

    return _router_probs(x, weight, scale, HIDDEN**-0.5, EPS)


def reference_pack(probs: torch.Tensor, cap: int):
    """``moe_union_pack`` on CPU in fp32: ``(slot_sticks [cap, STICK], routes [T, cap])``."""
    ids = torch.arange(EXPERTS, dtype=torch.float32)[:, None].contiguous()
    return _pack()(probs.float(), TOP_K, cap, ids, STICK)


def slot_scores(probs: torch.Tensor) -> torch.Tensor:
    """``moe_union_pack``'s per-expert slot score ``[E]`` (top-1 first, then mass)."""
    p = probs.float()
    values, index = p.topk(TOP_K, dim=-1)
    dense = torch.zeros_like(p).scatter(-1, index, values)
    dense = dense / dense.sum(-1, keepdim=True)
    top1 = (p >= p.amax(-1, keepdim=True)).float()
    return top1.amax(0) + dense.sum(0) / (p.shape[0] + 1)


def compare(ref_probs, cap, ref, actual) -> tuple[str, float, str]:
    """``(status, max route error, reason)`` of ``actual`` vs the fp32 ``ref``."""
    (ref_sticks, ref_routes), (sticks, routes) = ref, actual
    ref_ids, ids = ref_sticks[:, 0].round().long(), sticks[:, 0].round().long()
    if sorted(ref_ids.tolist()) != sorted(ids.tolist()):
        wrong = sorted(set(ids.tolist()) ^ set(ref_ids.tolist()))
        scores = slot_scores(ref_probs)
        cutoff = scores.sort(descending=True).values[cap - 1]
        near = (scores[wrong] - cutoff).abs() <= TIE_RTOL * cutoff
        status = "tie" if bool(near.all()) else "FAIL"
        return status, float("nan"), f"slot experts differ: {wrong}"
    order, ref_order = ids.argsort(), ref_ids.argsort()
    error = float(
        (routes.float()[:, order] - ref_routes.float()[:, ref_order]).abs().max()
    )
    if not error <= ROUTE_TOLERANCE:  # also catches NaN
        return "FAIL", error, f"routes off by {error:.4f}"
    return "ok", error, ""


def _failure_reason(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}"
    return next(
        (line.strip() for line in text.splitlines() if "DtException" in line),
        text.strip().splitlines()[0],
    )[:200]


def run_case(source: str, tokens: int, cap: int, device: str, verbose: bool) -> Result:
    pack = _pack()
    torch._dynamo.reset()
    seed = tokens * 1000 + cap

    def compiled(fn):
        if device == "cpu":
            return fn
        return torch.compile(fn, dynamic=False, fullgraph=True)

    def pack_region(p, ids):
        return pack(p, TOP_K, cap, ids, STICK)

    ids = torch.arange(EXPERTS, dtype=torch.float16)[:, None].contiguous()
    start = time.perf_counter()
    try:
        with torch.no_grad():
            if source == "router":
                inputs = router_inputs(tokens, seed)
                expected_probs = router_region(*(t.float() for t in inputs.values()))
                probs = compiled(router_region)(
                    *(t.to(device) for t in inputs.values())
                )
            else:
                probs = router_probs(tokens, seed).to(device)
                expected_probs = probs.to("cpu").float()
            sticks, routes = compiled(pack_region)(probs, ids.to(device))
            actual = (sticks.to("cpu"), routes.to("cpu"))
        seconds = time.perf_counter() - start
        # The pack is checked on the probabilities the device actually produced,
        # so router rounding cannot move slots; the router is checked on its own.
        pack_input = probs.to("cpu").float()
        probs_error = float((pack_input - expected_probs).abs().max())
        if not probs_error <= PROBS_TOLERANCE:
            return Result(
                source,
                tokens,
                cap,
                "FAIL",
                seconds,
                error=f"router probs off by {probs_error:.4f}",
            )
        ref = reference_pack(pack_input, cap)
        status, max_error, reason = compare(pack_input, cap, ref, actual)
        return Result(source, tokens, cap, status, seconds, max_error, reason)
    except Exception as error:  # report every failure, keep sweeping
        if verbose:
            traceback.print_exc()
        seconds = time.perf_counter() - start
        return Result(
            source, tokens, cap, "FAIL", seconds, error=_failure_reason(error)
        )


def format_results(results: list[Result]) -> str:
    lines = [
        f"{'probs':<6} {'T':>4} {'cap':>4}  {'result':<6} {'s':>7} "
        f"{'route err':>9}  error"
    ]
    for r in results:
        lines.append(
            f"{r.probs:<6} {r.tokens:>4} {r.cap:>4}  {r.status:<6} {r.seconds:>7.1f} "
            f"{r.max_route_error:>9.4f}  {r.error}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    parser.add_argument("--probs", nargs="+", choices=PROBS_SOURCES, default=["router"])
    parser.add_argument("--device", default=None, help="default: hf_common.DEVICE")
    parser.add_argument("-v", "--verbose", action="store_true", help="full tracebacks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for cap in args.caps:
        if not (max(args.tokens, TOP_K) <= cap <= EXPERTS):
            raise SystemExit(
                f"cap {cap} must be in [max(tokens, top_k), {EXPERTS}] "
                f"(tokens {args.tokens}, top_k {TOP_K})"
            )
    from hf_adapters import hf_common

    device = args.device or str(hf_common.DEVICE)
    print(f"device={device} tokens={args.tokens} experts={EXPERTS} top_k={TOP_K}")
    results = []
    for source in args.probs:
        for cap in args.caps:
            result = run_case(source, args.tokens, cap, device, args.verbose)
            print(f"[repro] probs={source} cap={cap}: {result.status}", flush=True)
            results.append(result)
    print()
    print(format_results(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
