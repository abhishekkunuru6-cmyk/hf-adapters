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

"""CPU tests for ``scripts/repro_moe_union_pack_caps.py`` (needs PR #721's
``hf_common.moe_union_pack``; skipped without it)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hf_adapters import hf_common

pytestmark = pytest.mark.skipif(
    not hasattr(hf_common, "moe_union_pack"),
    reason="needs PR #721's hf_common.moe_union_pack",
)

_SCRIPT = Path(__file__).parents[2] / "scripts" / "repro_moe_union_pack_caps.py"


@pytest.fixture(scope="module")
def repro():
    spec = importlib.util.spec_from_file_location("repro_moe_union_pack_caps", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def test_router_probs_are_a_softmax(repro):
    probs = repro.router_probs(16, seed=0)
    assert probs.shape == (16, repro.EXPERTS) and probs.dtype == torch.float16
    assert torch.allclose(probs.float().sum(-1), torch.ones(16), atol=1e-2)


@pytest.mark.parametrize("source", ["router", "host"])
@pytest.mark.parametrize("cap", [16, 24, 48])
def test_cpu_case_matches_its_reference(repro, source, cap):
    result = repro.run_case(source, 16, cap, "cpu", verbose=False)
    assert result.status == "ok", result.error
    assert result.max_route_error < 1e-2


def test_slot_scores_rank_like_moe_union_pack(repro):
    probs = repro.router_probs(16, seed=3)
    sticks, _ = repro.reference_pack(probs, 24)
    top = repro.slot_scores(probs).topk(24).indices
    assert sorted(top.tolist()) == sorted(sticks[:, 0].round().long().tolist())


def _swap_slot(repro, probs, cap, new_expert):
    ref = repro.reference_pack(probs, cap)
    sticks = ref[0].clone()
    sticks[-1] = new_expert  # replace one slot expert (column order is irrelevant)
    return ref, (sticks, ref[1])


def test_swap_at_the_cutoff_is_a_tie(repro):
    probs = repro.router_probs(16, seed=16 * 1000 + 28)  # the pod's cap-28 case
    order = repro.slot_scores(probs).sort(descending=True).indices
    ref, actual = _swap_slot(repro, probs, 28, order[28])  # 29th instead of 28th
    ref_sticks = ref[0].clone()
    ref_sticks[-1] = order[27]  # make sure the slot swapped out is the 28th
    status, _, reason = repro.compare(probs, 28, (ref_sticks, ref[1]), actual)
    assert status == "tie", reason


def test_swap_far_from_the_cutoff_fails(repro):
    probs = repro.router_probs(16, seed=0)
    order = repro.slot_scores(probs).sort(descending=True).indices
    ref, actual = _swap_slot(repro, probs, 16, order[-1])  # the weakest expert
    status, _, reason = repro.compare(probs, 16, ref, actual)
    assert status == "FAIL" and "slot" in reason


def test_cpu_sweep_reports_every_case(repro, capsys):
    argv = ["--device", "cpu", "--caps", "16", "24", "--probs", "router", "host"]
    assert repro.main(argv) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    cases = {
        (r[0], r[2]): r[3] for r in rows if len(r) > 3 and r[0] in ("router", "host")
    }
    assert cases == {(s, c): "ok" for s in ("router", "host") for c in ("16", "24")}


@pytest.mark.parametrize(
    "argv", [["--caps", "8"], ["--caps", "4", "--tokens", "2"], ["--caps", "256"]]
)
def test_rejects_invalid_caps(repro, argv):
    with pytest.raises(SystemExit):
        repro.main(["--device", "cpu", *argv])
