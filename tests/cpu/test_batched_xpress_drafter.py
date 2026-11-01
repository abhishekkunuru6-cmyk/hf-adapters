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


"""CPU tests for ``scripts/batched_xpress_drafter.py``: one drafter forward for a
batch must give each request what its own ``XPressTreeDrafter`` gives it."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("hf_adapters._xpress_tree")
helpers = pytest.importorskip("tests._gemma4_tree_helpers")

from hf_adapters import _xpress_tree  # noqa: E402

_SCRIPTS = Path(__file__).parents[2] / "scripts"
CPU = dict(dtype=torch.float32, device="cpu", compile=False)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bxd():
    return _load("batched_xpress_drafter")


def _tree(**shape):
    from tests.cpu.test_gemma4_spec_decode import _drafter

    return _xpress_tree.XPressTreeDrafter(_drafter(), **shape, **CPU)


def _width(d):
    return len(d.target_layer_ids) * d.fc[0].shape[0]


@pytest.mark.parametrize("depth", [7, 3], ids=["all-rows", "head-trimmed"])
def test_batched_draft_matches_each_request_alone(bxd, depth):
    """Prompts of different lengths (one past an append chunk and the sliding
    window), different new-row counts per step, a request going inactive."""
    shape = dict(frontier=1, branching=1, depth=depth)
    g = torch.Generator().manual_seed(3)
    lengths = (40, 140, 25)
    alone = [_tree(**shape) for _ in lengths]
    d = alone[0]
    W, block = _width(d), d.block
    prompts = [torch.randn(n, W, generator=g) for n in lengths]
    for a, p in zip(alone, prompts):
        a.reset(160 + 6 * block)
        a.append(p)
    bd = bxd.BatchedXPressDrafter(d, len(lengths))
    bd.reset(160, max_steps=6)
    bd.append_prompts(prompts)
    active = [True, True, True]
    for step in range(6):
        if step == 3:
            active[1] = False  # finished: writes nothing from now on
        anchors = [int(t) for t in torch.randint(5, 300, (3,), generator=g)]
        new = [
            None if step == 0 else torch.randn(int(m), W, generator=g)
            for m in torch.randint(1, block + 1, (3,), generator=g)
        ]
        outs = bd.draft(anchors, new, active)
        for b in range(3):
            if not active[b]:
                assert outs[b] is None
                continue
            want = alone[b].draft(anchors[b], new[b])
            # Head stats for slots 0..depth only; every row's hidden state.
            assert outs[b][0].shape[0] == bd.head_rows == depth + 1
            assert outs[b][2].shape[0] == block
            for got, ref in zip(outs[b], want):
                torch.testing.assert_close(
                    got, ref[: got.shape[0]], rtol=1e-4, atol=1e-4
                )
            tokens = bxd.search(d, outs[b], anchors[b], 7, depth + 1)
            assert tokens == bxd.search(d, want, anchors[b], 7, depth + 1)
    # Committed rows agree, including the request frozen when it went inactive.
    assert bd.rows == [a.rows for a in alone]
    assert "draft" not in vars(d)  # search removes its patch


def test_search_runs_the_beam_chain_through_the_same_patch(bxd):
    bsd = _load("batched_spec_decode")
    d = _tree(frontier=4, branching=4, depth=7)
    W = _width(d)
    d.reset(64)
    d.append(torch.randn(30, W, generator=torch.Generator().manual_seed(1)))
    outputs = d.draft(17)

    def beam(drafter, anchor, predecessor):
        return bsd.propose_chain_beam(
            drafter, anchor, predecessor, depth=7, beam=4, branching=4
        )

    got = bxd.search(d, outputs, 17, 9, 8, propose=beam)
    d.draft = lambda *a, **k: outputs
    try:
        want = beam(d, 17, 9)
    finally:
        del d.draft
    assert got == want


def test_validation(bxd):
    d = _tree(frontier=1, branching=1, depth=3)
    W, block = _width(d), d.block
    with pytest.raises(ValueError, match="batch"):
        bxd.BatchedXPressDrafter(d, 0)
    assert bxd.BatchedXPressDrafter(d, 1).head_rows == 4  # depth 3 + 1
    with pytest.raises(ValueError, match="head_rows"):
        bxd.BatchedXPressDrafter(d, 1, head_rows=block + 1)
    bd = bxd.BatchedXPressDrafter(d, 2)
    with pytest.raises(ValueError, match=">= 1"):
        bd.reset(0, 4)
    bd.reset(20, max_steps=1)
    with pytest.raises(ValueError, match="expected 2 prompts"):
        bd.append_prompts([torch.randn(5, W)])
    bd.append_prompts([torch.randn(5, W), torch.randn(9, W)])
    with pytest.raises(ValueError, match="at most"):
        bd.draft([5, 6], [torch.randn(block + 1, W), None])
    with pytest.raises(ValueError, match="expected 2"):
        bd.draft([5], [None])
    capacity = bd.capacity
    steps = (capacity - bd.pointer) // block
    for _ in range(steps):
        bd.draft([5, 6], [None, None])
    with pytest.raises(RuntimeError, match="cache full"):
        bd.draft([5, 6], [None, None])


@pytest.mark.parametrize(
    "shape",
    [
        dict(width=4, frontier=1, branching=1, depth=3),
        dict(width=8, frontier=3, branching=3, depth=5),
        dict(width=8, frontier=4, branching=4, depth=7, search="beam"),
    ],
    ids=["chain4", "tree8", "chain8-beam"],
)
def test_decoder_with_batched_drafter_matches_the_per_request_drafter(
    bxd, monkeypatch, shape
):
    """Same drafts -> same accepted lengths at every step, and greedy outputs."""
    from tests.cpu.test_gemma4_spec_decode import _drafter

    reference, adapted = helpers.tiny_gemma4_moe(monkeypatch)
    helpers.run_eagerly(monkeypatch)
    helpers._unwrap(adapted)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    bsd = _load("batched_spec_decode")
    bsd.configure_union_cap(adapted, None)
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(2, 320, (n,), generator=g) for n in (25, 38, 31)]

    def run(batched):
        decoder = bsd.BatchedSpecDecoder(
            adapted,
            _drafter(),
            shape,
            drafter_kwargs=CPU,
            batched_drafter=batched,
            draft_threads=1 if batched else 0,
        )
        return decoder.generate(prompts, max_new_tokens=18), decoder.stats

    outs, stats = run(True)
    ref_outs, ref_stats = run(False)
    assert outs == ref_outs
    assert stats.accepted == ref_stats.accepted
    for prompt, out in zip(prompts, outs):
        with torch.no_grad():
            greedy = reference.generate(
                prompt[None], max_new_tokens=18, do_sample=False, min_new_tokens=18
            )[0, prompt.numel() :]
        assert out == greedy.tolist()
