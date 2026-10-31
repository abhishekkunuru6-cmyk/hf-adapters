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

"""CPU tests for ``scripts/profile_xpress_draft_spyre.py`` (draft time split
into the drafter forward and the host tree search)."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("hf_adapters._xpress_tree")

_SCRIPTS = Path(__file__).parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prof():
    return _load("profile_xpress_draft_spyre")


def test_median(prof):
    assert prof.median_ms([0.003, 0.001, 0.002]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "shape",
    [
        dict(width=8, frontier=3, branching=3, depth=5),
        dict(width=8, frontier=1, branching=1, depth=7),
    ],
    ids=["tree8", "chain8"],
)
def test_profile_shape_splits_forward_and_search(prof, shape, monkeypatch):
    from tests.cpu.test_gemma4_spec_decode import _drafter

    from hf_adapters import _xpress_tree

    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    calls = []
    real = _xpress_tree.XPressTreeDrafter.draft

    def counting(self, *a, **kw):
        calls.append(kw.get("new_features", a[1] if len(a) > 1 else None))
        return real(self, *a, **kw)

    monkeypatch.setattr(_xpress_tree.XPressTreeDrafter, "draft", counting)
    result = prof.profile_shape(
        _drafter(),
        shape,
        steps=6,
        warmup=2,
        context=40,
        accepted=3,
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    assert len(calls) == 8  # warmup + measured, one forward per propose
    assert all(c is not None and c.shape[0] == 3 for c in calls)
    assert result["steps"] == 6 and result["nodes"] <= shape["width"]
    assert result["forward_ms"] > 0 and result["search_ms"] >= 0
    assert result["propose_ms"] == pytest.approx(
        result["forward_ms"] + result["search_ms"], rel=0.5
    )
    assert "draft" not in vars(result["drafter"])  # instance patch removed
