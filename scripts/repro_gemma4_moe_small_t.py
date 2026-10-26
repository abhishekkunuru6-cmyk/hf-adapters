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

"""Standalone repro: Gemma4 MoE multi-token routing fails to compile below 64 tokens.

The 16-token XPress verify of Gemma4 26B-A4B dies in ``_compiled_prefill_ffn`` with::

    InductorError: NotImplementedError: buf45 (SpyreReduction):
        no mechanism to resolve stick incompatibility

while 64 tokens compile. This script compiles the adapter's own multi-token MoE
graph on random weights with the real Gemma4 shapes (hidden 2816, 128 experts,
top-8, intermediate 704) at several token counts. No checkpoint, no model load.

Stages (each compiled on its own, ``dynamic=False``, ``fullgraph=True``):

- ``route``: ``hf_gemma4_moe._moe_route_persistent_packed`` — router softmax →
  top-k ``[T, 8]`` → ``spyre.keep_by_index`` → ``[T, 128]`` → normalise → stick pack.
- ``ffn``: ``route`` + ``hf_common.moe_prefill_all_experts`` (the all-experts loop),
  under the same named dims / LX-planning patch as the adapter.

Usage (Spyre pod, torch-spyre env, hf-adapters on PYTHONPATH)::

    python scripts/repro_gemma4_moe_small_t.py                       # T = 16 32 48 64
    python scripts/repro_gemma4_moe_small_t.py --tokens 16 --stage route -v

``--device cpu`` runs the same graphs eagerly (shape / numerics sanity check).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Gemma4 26B-A4B text config.
HIDDEN = 2816
EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 704
STICK = 64
EPS = 1e-6
DEFAULT_TOKENS = (16, 32, 48, 64)
STAGES = ("route", "ffn")


@dataclass(frozen=True)
class Result:
    stage: str
    tokens: int
    ok: bool
    seconds: float
    error: str = ""


def keep_by_index_reference(x, index, dim: int, fill: float):
    """CPU semantics of ``spyre.keep_by_index``: keep ``x`` at ``index`` along
    ``dim``, ``fill`` everywhere else."""
    mask = torch.zeros_like(x, dtype=torch.bool).scatter(dim, index, True)
    return torch.where(mask, x, torch.full_like(x, fill))


def ensure_keep_by_index() -> None:
    """Register a CPU ``spyre::keep_by_index`` when torch-spyre is not loaded."""
    if hasattr(torch.ops.spyre, "keep_by_index"):
        return
    lib = torch.library.Library("spyre", "FRAGMENT")
    lib.define("keep_by_index(Tensor x, Tensor index, int dim, float fill) -> Tensor")
    lib.impl("keep_by_index", keep_by_index_reference, "CPU")
    ensure_keep_by_index._lib = lib  # keep the registration alive


class _Experts(nn.Module):
    """HF-layout expert weights, as ``prepare_moe_expert_weights`` expects them."""

    def __init__(self, generator: torch.Generator, dtype: torch.dtype, experts: int):
        super().__init__()
        scale = HIDDEN**-0.5
        self.gate_up_proj = nn.Parameter(
            torch.randn(experts, 2 * INTERMEDIATE, HIDDEN, generator=generator)
            .mul(scale)
            .to(dtype),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.randn(experts, HIDDEN, INTERMEDIATE, generator=generator)
            .mul(INTERMEDIATE**-0.5)
            .to(dtype),
            requires_grad=False,
        )


def build_inputs(device: str, dtype: torch.dtype, experts: int, seed: int = 0) -> dict:
    """Random router + expert weights with Gemma4 shapes, on ``device``."""
    from hf_adapters import hf_common

    g = torch.Generator().manual_seed(seed)
    router = {
        "router_proj_w": (torch.randn(experts, HIDDEN, generator=g) * HIDDEN**-0.5)
        .to(dtype)
        .to(device),
        "router_scale": torch.ones(HIDDEN, dtype=dtype).to(device),
        "router_scalar_root_size": HIDDEN**-0.5,  # a Python float in HF
        "per_expert_scale": torch.ones(experts, dtype=dtype).to(device),
        "route_identity": torch.eye(STICK, dtype=dtype).to(device),
    }
    module = _Experts(g, dtype, experts)
    hf_common.prepare_moe_expert_weights(module, pad_to_multiple=STICK)
    return {
        **router,
        "gate": module.gate_proj,
        "up": module.up_proj,
        "down": module.down_proj,
    }


def route_fn(x, w: dict):
    from hf_adapters.hf_gemma4_moe import _moe_route_persistent_packed

    return _moe_route_persistent_packed(
        x,
        w["router_proj_w"],
        w["router_scale"],
        w["router_scalar_root_size"],
        w["per_expert_scale"],
        TOP_K,
        STICK,
        EPS,
        w["route_identity"],
    )


def ffn_fn(x, w: dict):
    from hf_adapters.hf_common import moe_prefill_all_experts

    routing_weight = route_fn(x, w)[..., :1]
    return moe_prefill_all_experts(
        x, routing_weight, w["gate"], w["up"], w["down"], "gelu_tanh"
    )


def run_stage(stage: str, tokens: int, w: dict, device: str, dtype, verbose: bool):
    from hf_adapters.hf_common import (
        named_moe_prefill_inputs,
        optional_spyre_config_patch,
    )

    torch._dynamo.reset()
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens))
    x = x.to(dtype).to(device)
    body = route_fn if stage == "route" else ffn_fn
    fn = body if device == "cpu" else torch.compile(body, dynamic=False, fullgraph=True)
    start = time.perf_counter()
    try:
        if stage == "ffn":
            with named_moe_prefill_inputs(x, w["gate"], w["up"], w["down"]):
                with optional_spyre_config_patch(
                    {"allow_all_ops_in_lx_planning": True}
                ):
                    out = fn(x, w)
        else:
            out = fn(x, w)
        out = out.cpu()
        if not torch.isfinite(out.float()).all():
            raise RuntimeError(f"non-finite output, shape {tuple(out.shape)}")
        return Result(stage, tokens, True, time.perf_counter() - start)
    except Exception as error:  # report every failure, keep sweeping
        if verbose:
            traceback.print_exc()
        first = f"{type(error).__name__}: {error}".strip().splitlines()
        return Result(stage, tokens, False, time.perf_counter() - start, first[0][:200])


def format_results(results: list[Result]) -> str:
    lines = [f"{'stage':<6} {'T':>4}  {'result':<6} {'s':>7}  error"]
    for r in results:
        status = "ok" if r.ok else "FAIL"
        lines.append(
            f"{r.stage:<6} {r.tokens:>4}  {status:<6} {r.seconds:>7.1f}  {r.error}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--device", default=None, help="default: hf_common.DEVICE")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--experts",
        type=int,
        default=EXPERTS,
        help="expert count (128 = Gemma4; fewer only to save memory)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="full tracebacks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(t < 2 for t in args.tokens):
        raise SystemExit("--tokens must be >= 2 (one token takes the decode path)")
    from hf_adapters import hf_common

    device = args.device or str(hf_common.DEVICE)
    if device == "cpu":
        hf_common.DEVICE = "cpu"
        ensure_keep_by_index()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    stages = STAGES if args.stage == "all" else (args.stage,)

    print(f"device={device} dtype={args.dtype} experts={args.experts} top_k={TOP_K}")
    weights = build_inputs(device, dtype, args.experts)
    results = []
    for stage in stages:
        for tokens in args.tokens:
            result = run_stage(stage, tokens, weights, device, dtype, args.verbose)
            print(
                f"[repro] {stage} T={tokens}: {'ok' if result.ok else 'FAIL'}",
                flush=True,
            )
            results.append(result)
    print()
    print(format_results(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
