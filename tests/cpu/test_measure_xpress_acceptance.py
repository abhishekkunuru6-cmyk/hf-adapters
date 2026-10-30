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

"""CPU tests for ``scripts/measure_xpress_acceptance.py``."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tests.cpu.test_xpress_cpu import BLOCK, HIDDEN, TAPS, _loaded_drafter

_SCRIPT = Path(__file__).parents[2] / "scripts" / "measure_xpress_acceptance.py"


@pytest.fixture(scope="module")
def acc():
    spec = importlib.util.spec_from_file_location("measure_xpress_acceptance", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draft, reference, expected",
    [
        ([1, 2, 3], [1, 2, 3], 3),
        ([1, 2, 9], [1, 2, 3], 2),
        ([9, 2, 3], [1, 2, 3], 0),
        ([1, 2, 3], [1, 2], 2),  # reference shorter than the draft
        ([], [1], 0),
    ],
)
def test_accepted_prefix(acc, draft, reference, expected):
    assert acc.accepted_prefix(draft, reference) == expected


def test_replay_advances_by_matches_plus_bonus(acc):
    tokens = list(range(100, 140))  # 40 tokens, prompt of 10
    calls = []

    def propose(anchor_pos):
        calls.append(anchor_pos)
        ref = tokens[anchor_pos + 1 : anchor_pos + 4]  # block 4 -> 3 drafts
        # Match 2 of 3 drafts, then a wrong token.
        return [ref[0], ref[1], -1]

    matches = acc.replay_blocks(tokens, first_anchor=10, block_size=4, propose=propose)
    # Each block accepts 2 drafts + 1 bonus token: anchors 10, 13, 16, ... while
    # a full block of reference tokens remains (anchor + 4 <= 40).
    assert calls == list(range(10, 37, 3))
    assert matches == [2] * len(calls)


def test_replay_rejects_wrong_draft_length(acc):
    with pytest.raises(ValueError, match="3 draft tokens"):
        acc.replay_blocks(
            list(range(20)), first_anchor=5, block_size=4, propose=lambda _: [1, 2]
        )


def test_replay_needs_a_full_block(acc):
    assert acc.replay_blocks(list(range(8)), 5, 4, propose=lambda _: [0] * 3) == []


def test_replay_caps_verified_drafts(acc):
    """verify_tokens=3 checks only the first 2 drafts: a block yields at most 3."""
    tokens = list(range(100, 140))

    def propose(anchor_pos):
        # 3 drafts, all correct (padded past the end of the continuation).
        return (tokens[anchor_pos + 1 : anchor_pos + 4] + [-1] * 3)[:3]

    matches = acc.replay_blocks(tokens, 10, 4, propose=propose, verify_tokens=3)
    assert set(matches) == {2}
    # Anchors advance by 3 while a 3-token verify block remains (anchor + 3 <= 40).
    assert len(matches) == len(range(10, 38, 3))


@pytest.mark.parametrize("verify_tokens", [1, 5])
def test_replay_rejects_bad_verify_tokens(acc, verify_tokens):
    with pytest.raises(ValueError, match="verify_tokens"):
        acc.replay_blocks(
            list(range(20)),
            5,
            4,
            propose=lambda _: [0] * 3,
            verify_tokens=verify_tokens,
        )


def test_verify_ratio_prefers_per_k_costs(acc):
    assert acc.verify_ratio(8, {"8": 2.5}, fallback=12.6) == 2.5
    assert acc.verify_ratio(4, {"8": 2.5}, fallback=12.6) == 12.6
    assert acc.verify_ratio(4, None, fallback=12.6) == 12.6
    assert acc.verify_ratio(4, {"8": 2.5}, fallback=None) is None


def test_summarize(acc):
    stats = acc.summarize([0, 3, 1, 3], block_size=4)
    assert stats.blocks == 4
    assert stats.mean_matches == pytest.approx(1.75)
    assert stats.tokens_per_block == pytest.approx(2.75)
    assert stats.full_blocks == 2


def test_summarize_requires_blocks(acc):
    with pytest.raises(ValueError, match="no blocks"):
        acc.summarize([], block_size=4)


def test_projected_speedup(acc):
    # 6 tokens per block, verify 2x decode, draft 1x decode -> 6 / 3.
    assert acc.projected_speedup(6.0, 2.0, 1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        acc.projected_speedup(6.0, 0.0, 0.0)


def test_tap_indices(acc):
    assert acc.tap_indices([1, 6, 27], offset=1) == [2, 7, 28]
    assert acc.tap_indices([1, 6], offset=0) == [1, 6]
    with pytest.raises(ValueError, match="offset"):
        acc.tap_indices([0], offset=-1)


def test_config_grid_and_labels(acc):
    grid = acc.config_grid([0, 6], [512], ["anchor", "real"], [48])
    assert len(grid) == 4
    assert grid[0] == acc.ProposeConfig(
        num_passes=0, topc=512, predecessor="anchor", ctx_pad=48
    )
    assert grid[-1].label() == "K=6 topc=512 pred=real ctx=48"
    with pytest.raises(ValueError, match="predecessor"):
        acc.config_grid([6], [512], ["bogus"], [48])


def test_chunk_winners_keeps_each_chunks_first_max(acc):
    # 10 ids in chunks of 4: [0..3], [4..7], [8, 9] (+2 padding).
    base = torch.tensor([[[1.0, 5.0, 5.0, 0.0, 2.0, 1.0, 3.0, 9.0, 4.0, 4.5]]])
    values, ids = acc.chunk_winners(base, chunk=4)
    assert ids.tolist() == [[[1, 7, 9]]]  # first index on the 5.0 tie
    assert values.tolist() == [[[5.0, 9.0, 4.5]]]
    assert torch.equal(base.gather(-1, ids), values)


def test_chunk_winners_always_hold_the_global_argmax(acc):
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 15, 1000, generator=g)  # 1000 is not a 64 multiple
    values, ids = acc.chunk_winners(base)
    assert ids.shape == (1, 15, 16)
    assert torch.equal(
        ids.gather(-1, values.argmax(-1, keepdim=True))[..., 0], base.argmax(-1)
    )


def test_config_grid_with_candidate_rules(acc):
    grid = acc.config_grid(
        [6], [512, 4096], ["real"], [1024], candidates=["topc", "chunk"]
    )
    # topc is per rule only for "topc"; "chunk" has one config per setting.
    assert [(c.candidates, c.topc) for c in grid] == [
        ("topc", 512),
        ("topc", 4096),
        ("chunk", 0),
    ]
    assert grid[0].label() == "K=6 topc=512 pred=real ctx=1024"
    assert grid[-1].label() == "K=6 cand=chunk64 pred=real ctx=1024"
    with pytest.raises(ValueError, match="candidate"):
        acc.config_grid([6], [512], ["real"], [1024], candidates=["bogus"])


def test_config_grid_with_verify_tokens(acc):
    grid = acc.config_grid([6], [512], ["real"], [1024], verify_tokens=[4, 16])
    assert [c.verify_tokens for c in grid] == [4, 16]
    assert grid[0].label() == "K=6 topc=512 pred=real ctx=1024 v=4"


# ---------------------------------------------------------------------------
# Drafter wiring, on the tiny random drafter of test_xpress_cpu
# ---------------------------------------------------------------------------


def test_force_cpu_device_patches_every_adapter_module(acc, monkeypatch):
    """Off the pytest DEVICE patch, every module that moves tensors (incl. the
    RoPE cache in hf_common) must target CPU, or the drafter mixes devices."""
    from hf_adapters import _dspark_common, hf_common, hf_xpress_qwen3

    modules = (hf_common, _dspark_common, hf_xpress_qwen3)
    for module in modules:
        monkeypatch.setattr(module, "DEVICE", "spyre:0")
    acc.force_cpu_device()
    assert all(module.DEVICE == "cpu" for module in modules)


def _tiny_record(seq_len: int = 30, prompt_len: int = 12):
    g = torch.Generator().manual_seed(7)
    return {
        "input_ids": torch.randint(0, 1000, (seq_len,), generator=g),
        "prompt_len": prompt_len,
        "taps": torch.randn(seq_len, TAPS * HIDDEN, generator=g),
    }


@pytest.mark.parametrize(
    "predecessor, candidates", [("anchor", "topc"), ("real", "topc"), ("real", "chunk")]
)
def test_measure_record_on_its_own_drafts_accepts_everything(
    acc, predecessor, candidates
):
    """A reference built from the drafter's own proposals is fully accepted, so
    the replay, the context slicing and the predecessor wiring line up."""
    from hf_adapters import hf_xpress_qwen3 as xp

    cfg = acc.ProposeConfig(
        num_passes=2,
        topc=0,
        predecessor=predecessor,
        ctx_pad=16,
        candidates=candidates,
    )
    model = acc.prepare_cpu_drafter(_loaded_drafter(xp), cfg.ctx_pad)
    record = _tiny_record()
    tokens = record["input_ids"].tolist()
    prompt_len = record["prompt_len"]
    # Rewrite the "target" continuation block by block with the drafter's own
    # drafts (+ one arbitrary bonus token), the way an ideal target would.
    anchor = prompt_len
    while anchor + BLOCK <= len(tokens):
        draft = acc.propose_at(model, record["taps"], tokens, anchor, cfg)
        tokens[anchor + 1 : anchor + BLOCK] = draft
        anchor += BLOCK
    record = {**record, "input_ids": torch.tensor(tokens)}

    matches = acc.measure_record(model, record, cfg)
    assert matches and all(m == BLOCK - 1 for m in matches)
