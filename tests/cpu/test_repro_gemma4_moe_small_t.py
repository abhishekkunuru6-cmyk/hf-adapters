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

"""CPU tests for ``scripts/repro_gemma4_moe_small_t.py``."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_SCRIPT = Path(__file__).parents[2] / "scripts" / "repro_gemma4_moe_small_t.py"
EXPERTS = 8  # small, so the CPU run stays fast


@pytest.fixture(scope="module")
def repro():
    spec = importlib.util.spec_from_file_location("repro_gemma4_moe_small_t", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    module.ensure_keep_by_index()
    return module


@pytest.fixture(scope="module")
def weights(repro):
    return repro.build_inputs("cpu", torch.float32, EXPERTS)


def test_keep_by_index_reference(repro):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = repro.keep_by_index_reference(x, torch.tensor([[3, 1]]), -1, 0.0)
    assert out.tolist() == [[0.0, 2.0, 0.0, 4.0]]


def _router_reference(x, w, top_k):
    """HF Gemma4 router: RMS norm, scale, softmax, top-k, renormalise."""
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    logits = F.linear(
        normed * w["router_scale"] * w["router_scalar_root_size"], w["router_proj_w"]
    )
    probs = logits.softmax(-1)
    top, idx = probs.topk(top_k, dim=-1)
    dense = torch.zeros_like(probs).scatter(-1, idx, top / top.sum(-1, keepdim=True))
    return dense * w["per_expert_scale"]


@pytest.mark.parametrize("tokens", [16, 64])
def test_route_matches_router_reference(repro, weights, tokens):
    x = torch.randn(tokens, repro.HIDDEN)
    packed = repro.route_fn(x, weights)
    assert packed.shape == (tokens, EXPERTS, repro.STICK)
    expected = _router_reference(x, weights, repro.TOP_K)
    torch.testing.assert_close(packed[..., 0], expected, rtol=1e-4, atol=1e-5)


def test_ffn_matches_dense_reference(repro, weights):
    x = torch.randn(16, repro.HIDDEN)
    route = _router_reference(x, weights, repro.TOP_K)  # [T, E]
    gate, up, down = weights["gate"], weights["up"], weights["down"]
    act = F.gelu(torch.einsum("th,ehm->etm", x, gate), approximate="tanh")
    per_expert = torch.einsum(
        "etm,emh->eth", act * torch.einsum("th,ehm->etm", x, up), down
    )
    expected = (per_expert * route.T.unsqueeze(-1)).sum(0)
    torch.testing.assert_close(repro.ffn_fn(x, weights), expected, rtol=1e-4, atol=1e-4)


def test_cpu_sweep_reports_every_stage(repro, capsys):
    assert repro.main(["--device", "cpu", "--experts", "8", "--tokens", "16"]) == 0
    out = capsys.readouterr().out
    assert "route    16  ok" in out and "ffn      16  ok" in out


def test_rejects_single_token(repro):
    with pytest.raises(SystemExit):
        repro.main(["--device", "cpu", "--tokens", "1"])
