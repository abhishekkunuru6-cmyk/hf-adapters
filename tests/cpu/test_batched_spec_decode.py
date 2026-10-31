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

"""CPU tests for ``scripts/batched_spec_decode.py`` (batched verify on #721).

A tiny random Gemma 4 MoE through the real adapter preparation (#721's test
helpers): every request of a batch must score each tree path as the stock HF
model does on that path alone, across commits with holes and past the
sliding window, and batched speculative decoding must reproduce each
prompt's greedy continuation. Skipped without #721's modules.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("hf_adapters._gemma4_tree")
helpers = pytest.importorskip("tests._gemma4_tree_helpers")

from hf_adapters._gemma4_tree import global_tree_mask, tree_ancestors  # noqa: E402

from hf_adapters import _gemma4_tree, _xpress_tree  # noqa: E402

_SCRIPTS = Path(__file__).parents[2] / "scripts"
VOCAB = 320


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bsd():
    return _load("batched_spec_decode")


@pytest.fixture
def models(monkeypatch):
    reference, adapted = helpers.tiny_gemma4_moe(monkeypatch)
    helpers.run_eagerly(monkeypatch)
    helpers._unwrap(adapted)
    return reference, adapted


def _prompts(*lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (n,), generator=g) for n in lengths]


# -- host helpers ---------------------------------------------------------------


def test_global_mask_matches_721_for_one_contiguous_request(bsd):
    ancestors = tree_ancestors(helpers.PARENTS)
    start, valid_start, capacity = 70, 6, 128
    row_pos = torch.full((1, capacity), -1)
    row_pos[0, valid_start:start] = torch.arange(start - valid_start)
    depths = _gemma4_tree.tree_depths(helpers.PARENTS)
    positions = torch.tensor([[start - valid_start + d for d in depths]])
    full, _ = bsd.batch_masks(row_pos, positions, [ancestors], start, 64, torch.float32)
    want = global_tree_mask(ancestors, start, valid_start, capacity, torch.float32)
    torch.testing.assert_close(full, want)


def test_masks_skip_holes_and_keep_exactly_the_window(bsd):
    row_pos = torch.full((2, 16), -1)
    row_pos[0, :6] = torch.tensor([0, 1, -1, 2, -1, 3])  # rows 2, 4: rejected
    row_pos[1, 2:6] = torch.arange(4)  # left padding
    positions = torch.tensor([[4, 5], [4, 5]])
    ancestors = [[[0], [0, 1]]] * 2
    full, slid = bsd.batch_masks(row_pos, positions, ancestors, 6, 3, torch.float32)

    def allowed(mask, b, j):
        return (mask[b, 0, j] == 0).nonzero().flatten().tolist()

    assert allowed(full, 0, 0) == [0, 1, 3, 5, 6]
    assert allowed(full, 1, 1) == [2, 3, 4, 5, 6, 7]
    # Window 3: node at position 4 reads positions 2, 3 (+ itself).
    assert allowed(slid, 0, 0) == [3, 5, 6]
    assert allowed(slid, 0, 1) == [5, 6, 7]


def test_pad_tree_matches_721(bsd):
    from hf_adapters.gemma4_spec_decode import TreeSpeculativeDecoder

    shim = type("Shim", (), {"width": 8})()
    tokens, parents = [5, 0, 7, 1], [-1, 0, 1, 0]
    want = TreeSpeculativeDecoder._pad_tree(shim, tokens, parents)
    assert bsd.pad_tree(tokens, parents, 8) == want


def test_configure_union_cap(bsd, models):
    _, adapted = models
    assert bsd.configure_union_cap(adapted, 12) == 12
    block = _gemma4_tree._backbone(adapted).layers[0]
    assert block._moe_union_cap == 12 and block.router.slot_identity.shape == (12, 12)
    assert bsd.configure_union_cap(adapted, 99) == 16  # clamped to the experts
    assert bsd.configure_union_cap(adapted, None) is None
    with pytest.raises(ValueError, match="top_k"):
        bsd.configure_union_cap(adapted, 2)


# -- batched verify against the reference ---------------------------------------


@pytest.mark.parametrize("cap", [None, 16], ids=["cap-off", "cap-all-experts"])
def test_batched_verify_matches_the_reference_on_every_path(bsd, models, cap):
    reference, adapted = models
    bsd.configure_union_cap(adapted, cap)
    width = 8 if cap else len(helpers.PARENTS)  # B * width <= cap: lossless
    parents = helpers.PARENTS[:width]
    prompts = _prompts(40, 23)
    verifier = bsd.BatchedTreeVerifier(adapted, width, prefill_chunk=64)
    logits = verifier.prefill(prompts, max_steps=4)
    g = torch.Generator().manual_seed(1)
    trees = []
    for b, prompt in enumerate(prompts):
        want = helpers._reference_next(reference, prompt.tolist())
        torch.testing.assert_close(logits[b, -1, :VOCAB], want, rtol=1e-4, atol=1e-4)
        rest = torch.randint(0, VOCAB, (width - 1,), generator=g).tolist()
        trees.append(([int(want.argmax())] + rest, parents))
    logits = verifier.verify(trees)
    for b, (tokens, _) in enumerate(trees):
        for node, path in enumerate(tree_ancestors(parents)):
            history = prompts[b].tolist() + [tokens[p] for p in path]
            want = helpers._reference_next(reference, history)
            torch.testing.assert_close(
                logits[b, node, :VOCAB], want, rtol=1e-4, atol=1e-4
            )


def test_commits_with_holes_continue_like_plain_decoding(bsd, models):
    """Different off-spine paths per request for 24 steps, past the 64 window."""
    reference, adapted = models
    bsd.configure_union_cap(adapted, None)
    parents = helpers.PARENTS
    prompts = _prompts(50, 31, seed=3)
    verifier = bsd.BatchedTreeVerifier(adapted, len(parents), prefill_chunk=64)
    verifier.prefill(prompts, max_steps=24)
    history = [p.tolist() for p in prompts]
    anchors = [int(helpers._reference_next(reference, h).argmax()) for h in history]
    choices = [[0, 6, 8, 11], [0, 1, 2, 3], [0], [0, 1, 7]]
    g = torch.Generator().manual_seed(5)
    for step in range(24):
        trees = [
            (
                [a]
                + torch.randint(0, VOCAB, (len(parents) - 1,), generator=g).tolist(),
                parents,
            )
            for a in anchors
        ]
        logits = verifier.verify(trees)
        paths = [choices[(step + b) % 4] for b in range(len(prompts))]
        for b, path in enumerate(paths):
            history[b] += [trees[b][0][p] for p in path]
            want = helpers._reference_next(reference, history[b])
            torch.testing.assert_close(
                logits[b, path[-1], :VOCAB], want, rtol=1e-4, atol=1e-4
            )
            anchors[b] = int(want.argmax())
        verifier.commit(paths)
    assert min(verifier.state.lengths) > 64 + 31


def test_verify_rejects_a_full_cache_and_double_verify(bsd, models):
    _, adapted = models
    bsd.configure_union_cap(adapted, None)
    verifier = bsd.BatchedTreeVerifier(adapted, 8, prefill_chunk=64)
    verifier.prefill(_prompts(20, 20), max_steps=1)
    tree = ([1] * 8, [-1] + [0] * 7)
    with pytest.raises(ValueError, match="one tree"):
        verifier.verify([tree])
    tree = (list(range(8)), [-1] + [0] * 7)
    verifier.verify([tree, tree])
    with pytest.raises(RuntimeError, match="twice"):
        verifier.verify([tree, tree])
    verifier.commit([[0], None])
    verifier.state.pointer = verifier.state.capacity - 4
    with pytest.raises(RuntimeError, match="full"):
        verifier.verify([tree, tree])


# -- end to end -----------------------------------------------------------------


@pytest.mark.parametrize("shape", ["tree8", "chain8"])
def test_batched_speculative_decoding_equals_greedy(bsd, models, monkeypatch, shape):
    from tests.cpu.test_gemma4_spec_decode import _drafter

    reference, adapted = models
    bsd.configure_union_cap(adapted, None)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    shapes = {
        "tree8": dict(width=8, frontier=3, branching=3, depth=5),
        "chain8": dict(width=8, frontier=1, branching=1, depth=7),
    }
    prompts = _prompts(30, 44, 17, seed=4)
    decoder = bsd.BatchedSpecDecoder(
        adapted,
        _drafter(),
        shapes[shape],
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    outs = decoder.generate(prompts, max_new_tokens=30)
    for prompt, out in zip(prompts, outs):
        with torch.no_grad():
            greedy = reference.generate(
                prompt[None], max_new_tokens=30, do_sample=False, min_new_tokens=30
            )[0, prompt.numel() :]
        assert out == greedy.tolist()
    stats = decoder.stats
    assert stats.batch == 3 and len(stats.verify_ms) == stats.steps()
    assert sum(stats.emitted()) >= 3 * 29


def test_drafter_cache_size_does_not_follow_the_prompt_length(bsd, models, monkeypatch):
    """Groups whose prompts share a prefill chunk must reuse the drafter's graphs
    (a new cache size recompiles them inside the measured steps)."""
    from tests.cpu.test_gemma4_spec_decode import _drafter

    _, adapted = models
    bsd.configure_union_cap(adapted, None)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    decoder = bsd.BatchedSpecDecoder(
        adapted,
        _drafter(),
        dict(width=8, frontier=1, branching=1, depth=7),
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    sizes = []
    for lengths in [(12, 20), (40, 55)]:  # all within one 512-row prefill chunk
        decoder.generate(_prompts(*lengths), max_new_tokens=6)
        sizes.append({s["capacity"] for s in decoder.slots.states})
    assert sizes[0] == sizes[1] and len(sizes[0]) == 1


# -- round 1: chain length, beam chain, device argmax, readback timing ---------


def test_new_shapes(bsd):
    assert bsd.SHAPES["chain4"] == dict(width=4, frontier=1, branching=1, depth=3)
    assert bsd.SHAPES["chain6"] == dict(width=6, frontier=1, branching=1, depth=5)
    beam = bsd.SHAPES["chain8-beam"]
    assert beam["width"] == 8 and beam["depth"] == 7 and beam["search"] == "beam"


def test_chunk_argmax_matches_argmax_with_ties_and_padding(bsd):
    g = torch.Generator().manual_seed(0)
    vocab, padded = 300, 448  # 7 chunks of 64; columns >= 300 are padding
    logits = torch.randn(2, 3, padded, generator=g).half()
    logits[0, 0, 10] = logits[0, 0, 200] = 9.0  # tie across chunks: first wins
    logits[0, 1, 70] = logits[0, 1, 75] = 9.0  # tie inside a chunk
    logits[1, 2, 350] = 50.0  # a padded column must never win
    bias = bsd.vocab_bias(vocab, padded, torch.float16)
    positions = torch.arange(bsd.CHUNK, dtype=torch.float16)
    for slice_cols in (128, 448, 4096):  # several slices, one, larger than the row
        stats = bsd.chunk_argmax_stats(logits, bias, positions, slice_cols)
        ids = bsd.decode_chunk_argmax(stats.float(), padded // bsd.CHUNK)
        want = logits[:, :, :vocab].float().argmax(-1).flatten()
        assert ids.tolist() == want.tolist()
    assert ids[0] == 10 and ids[1] == 70


def test_greedy_ids_on_the_device_equals_the_host(bsd, models):
    _, adapted = models
    bsd.configure_union_cap(adapted, None)
    host = bsd.BatchedTreeVerifier(adapted, 8, prefill_chunk=64)
    dev = bsd.BatchedTreeVerifier(adapted, 8, prefill_chunk=64, device_argmax=True)
    prompts = _prompts(20, 31, seed=8)
    host.prefill(prompts, max_steps=2)
    tree = (list(range(8)), [-1, 0, 1, 2, 3, 4, 5, 6])
    logits = host.verify([tree, tree])
    assert dev.greedy_ids(logits).tolist() == host.greedy_ids(logits).tolist()
    assert host.greedy_ids(logits).shape == (2, 8)


def _drafter_pair(seed=0):
    from tests.cpu.test_gemma4_spec_decode import _drafter

    model = _drafter(seed)
    kw = dict(dtype=torch.float32, device="cpu", compile=False)
    return model, kw


def _primed(model, kw, options, features):
    d = _xpress_tree.XPressTreeDrafter(model, **options, **kw)
    d.reset(200)
    d.append(features)
    return d


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beam_chain_with_beam_one_is_the_greedy_chain(bsd, monkeypatch, seed):
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    model, kw = _drafter_pair(seed)
    g = torch.Generator().manual_seed(seed)
    width = len(model.target_layer_ids) * model.config.hidden_size
    ctx, new = torch.randn(30, width, generator=g), torch.randn(3, width, generator=g)
    chain = dict(frontier=1, branching=1, depth=7)
    a = _primed(model, kw, chain, ctx)
    b = _primed(model, kw, chain, ctx)
    want = a.propose(17, 5, 8, new_features=new)
    got = bsd.propose_chain_beam(
        b, 17, 5, depth=7, beam=1, branching=1, new_features=new
    )
    assert got == want
    assert got[1] == [-1, 0, 1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beam_chain_maximizes_expected_accepted_length(bsd, monkeypatch, seed):
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    model, kw = _drafter_pair(seed)
    g = torch.Generator().manual_seed(10 + seed)
    width = len(model.target_layer_ids) * model.config.hidden_size
    ctx, new = torch.randn(30, width, generator=g), torch.randn(3, width, generator=g)
    chain = dict(frontier=1, branching=1, depth=7)
    greedy = _primed(model, kw, chain, ctx).propose(17, 5, 8, new_features=new)[0]
    beam, parents = bsd.propose_chain_beam(
        _primed(model, kw, chain, ctx),
        17,
        5,
        depth=7,
        beam=4,
        branching=4,
        new_features=new,
    )
    assert parents == list(range(-1, len(beam) - 1)) and beam[0] == 17
    ref = _primed(model, kw, chain, ctx)
    cmax, first, hidden = ref.draft(17, new)

    def expected(tokens):  # sum over prefixes of P(prefix), from #721's reference
        return sum(
            math.exp(
                _xpress_tree.path_logprob(
                    ref, hidden, cmax, first, 5, 17, tokens[1 : k + 1]
                )
            )
            for k in range(1, len(tokens))
        )

    assert expected(beam) >= expected(greedy) - 1e-6


@pytest.mark.parametrize("shape", ["chain4", "chain8-beam"])
def test_new_shapes_decode_greedy(bsd, models, monkeypatch, shape):
    from tests.cpu.test_gemma4_spec_decode import _drafter

    reference, adapted = models
    bsd.configure_union_cap(adapted, None)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    spec = dict(bsd.SHAPES[shape])
    if spec.get("search") == "beam":
        spec["depth"] = 7  # the tiny drafter's block is 8
    prompts = _prompts(25, 38, seed=9)
    decoder = bsd.BatchedSpecDecoder(
        adapted,
        _drafter(),
        spec,
        device_argmax=True,
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    outs = decoder.generate(prompts, max_new_tokens=20)
    for prompt, out in zip(prompts, outs):
        with torch.no_grad():
            greedy = reference.generate(
                prompt[None], max_new_tokens=20, do_sample=False, min_new_tokens=20
            )[0, prompt.numel() :]
        assert out == greedy.tolist()
    stats = decoder.stats
    assert len(stats.readback_ms) == stats.steps()
    assert all(r <= v for r, v in zip(stats.readback_ms, stats.verify_ms))
