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

"""CPU tests for ``scripts/bench_gemma4_batched_spec_spyre.py`` (chain vs tree
at batch sizes x union caps)."""

import importlib.util
import types
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
def bench():
    return _load("bench_gemma4_batched_spec_spyre")


def _stats(batch, accepted, verify, draft, host):
    s = types.SimpleNamespace(
        batch=batch, accepted=accepted, verify_ms=verify, draft_ms=draft, host_ms=host
    )
    s.emitted = lambda: [a for step in accepted for a in step if a is not None]
    s.steps = lambda: len(accepted)
    return s


def test_parse_cap(bench):
    assert bench.parse_cap("off") is None and bench.parse_cap("None") is None
    assert bench.parse_cap("32") == 32
    with pytest.raises(ValueError):
        bench.parse_cap("0")
    with pytest.raises(ValueError):
        bench.parse_cap("many")


def test_prompt_groups(bench):
    assert bench.prompt_groups(list(range(8)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert bench.prompt_groups(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(ValueError, match="at least"):
        bench.prompt_groups([0, 1], 4)


def test_first_difference(bench):
    assert bench.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert bench.first_difference([1, 2, 3], [1, 5, 3]) == 1
    assert bench.first_difference([1, 2], [1, 2, 3]) == 2


def test_summarize_counts_tokens_per_request_and_time(bench):
    # Two batch-2 runs; request 1 of the first run finished after step 1.
    runs = [
        _stats(2, [[4, 2], [3, None]], [100.0, 120.0], [40.0, 20.0], [5.0, 5.0]),
        _stats(2, [[5, 5]], [110.0], [30.0], [4.0]),
    ]
    row = bench.summarize(
        "chain8", 8, 2, 16, runs, [True, False, True, True], [None, 7, None, None]
    )
    assert row.rows == 16 and row.covers_top1 and row.status == "ok"
    assert row.tokens_per_step == pytest.approx(19 / 5, abs=1e-3)
    assert row.verify_ms == 110.0
    assert row.draft_ms == 15.0  # medians of 20, 10, 15 (per request)
    wall = 100 + 120 + 110 + 40 + 20 + 30 + 5 + 5 + 4
    assert row.tokens_per_s == pytest.approx(1000 * 19 / wall, abs=0.01)
    assert row.verify_tokens_per_s == pytest.approx(1000 * 19 / 330, abs=0.01)
    assert row.steps == 3 and row.requests == 4
    assert row.match_ar == 0.75 and row.median_first_diff == 7


def test_rows_above_the_cap_are_flagged(bench):
    row = bench.summarize(
        "tree16", 16, 4, 32, [_stats(4, [[1] * 4], [1.0], [1.0], [1.0])]
    )
    assert row.rows == 64 and not row.covers_top1 and row.match_ar is None
    assert bench.summarize("tree16", 16, 8, None, []).covers_top1  # cap off


def test_failed_row_and_table(bench):
    ok = bench.summarize(
        "tree16", 16, 1, 16, [_stats(1, [[4]], [130.0], [40.0], [10.0])], [True], [None]
    )
    bad = bench.failed("chain8", 8, 4, 24, "DtException: Incorrect chunk size")
    table = bench.format_table([ok, bad])
    lines = table.splitlines()
    assert len(lines) == 4 and lines[0].startswith("| shape | batch | cap")
    assert "| tree16 | 1 | 16 | 16 | yes | 4.00 | 130.0 | 40.0 |" in lines[2]
    assert "| chain8 | 4 | 24 | 32 | no |" in lines[3]
    assert "FAIL: DtException: Incorrect chunk size" in lines[3]


def test_run_config_on_a_tiny_model(bench, monkeypatch):
    """The driver end to end on CPU: warmup + measured groups, greedy match."""
    pytest.importorskip("hf_adapters._gemma4_tree")
    helpers = pytest.importorskip("tests._gemma4_tree_helpers")
    from tests.cpu.test_gemma4_spec_decode import _drafter

    from hf_adapters import _xpress_tree

    reference, adapted = helpers.tiny_gemma4_moe(monkeypatch)
    helpers.run_eagerly(monkeypatch)
    helpers._unwrap(adapted)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    bsd = _load("batched_spec_decode")
    monkeypatch.setitem(
        bsd.SHAPES, "chain4", dict(width=4, frontier=1, branching=1, depth=3)
    )
    drafter = _drafter()
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 320, (n,), generator=g) for n in (20, 33, 26, 41)]
    refs = [
        reference.generate(
            p[None], max_new_tokens=12, do_sample=False, min_new_tokens=12
        )[0, p.numel() :].tolist()
        for p in prompts
    ]
    tree_drafter = _xpress_tree.XPressTreeDrafter(
        drafter,
        frontier=1,
        branching=1,
        depth=3,
        dtype=torch.float32,
        device="cpu",
        compile=False,
    )
    args = types.SimpleNamespace(
        max_new_tokens=12,
        max_steps=12,
        eos=None,
        tree_drafters={"chain4": tree_drafter},
    )
    # Cap 16 = all 16 experts of the tiny model: lossless, so greedy holds.
    row = bench.run_config(bsd, adapted, drafter, "chain4", 2, 16, prompts, refs, args)
    assert row.status == "ok" and row.requests == 4 and row.match_ar == 1.0
    assert row.rows == 8 and row.covers_top1 and row.tokens_per_step >= 1.0
    assert args.tree_drafters["chain4"] is tree_drafter
