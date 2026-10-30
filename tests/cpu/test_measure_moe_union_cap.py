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

"""CPU tests for the union-cap simulation in ``scripts/measure_moe_expert_usage.py``
(``cap-report``: how much of a verify's expert set ``SPYRE_MOE_UNION_CAP`` keeps)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from hf_adapters import hf_common

_SCRIPT = Path(__file__).parents[2] / "scripts" / "measure_moe_expert_usage.py"


@pytest.fixture(scope="module")
def usage():
    spec = importlib.util.spec_from_file_location("measure_moe_expert_usage", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probs(rows):
    """``[1, T, E]`` router probabilities from unnormalized rows."""
    p = torch.tensor(rows, dtype=torch.float32)
    return (p / p.sum(dim=-1, keepdim=True))[None]


# 3 tokens, 6 experts, top-2. Top-1s: 0, 0, 5. Union {0, 1, 2, 3, 5}.
# Renormalized top-2 mass per expert: 0 -> 0.6 + 5/9, 1 -> 0.4,
# 2 -> 4/9, 3 -> 0.4, 5 -> 0.6.
_ROWS = [
    [6, 4, 0, 0, 0, 0],
    [5, 0, 4, 0, 0, 0.01],
    [0, 0, 0, 4, 0, 6],
]


def test_slots_put_every_top1_first(usage):
    # Expert 5 is a top-1 with less mass than expert 0 but it must get a slot
    # before expert 2 (mass 4/9), which is nobody's top-1.
    slots = usage.union_slot_mask(_probs(_ROWS), top_k=2, cap=2)
    assert slots[0].nonzero().flatten().tolist() == [0, 5]


def test_slots_then_take_most_routed_mass(usage):
    slots = usage.union_slot_mask(_probs(_ROWS), top_k=2, cap=3)
    assert slots[0].nonzero().flatten().tolist() == [0, 2, 5]


def test_cap_stats_drop_counts_and_mass(usage):
    s = usage.cap_window_stats(_probs(_ROWS), top_k=2, cap=3)
    assert s.union.tolist() == [5]
    assert s.kept.tolist() == [3]
    # Token 0 loses expert 1 (0.4); token 1 keeps 0 and 2; token 2 loses 3 (0.4).
    assert torch.allclose(s.dropped_mass, torch.tensor([[0.4, 0.0, 0.4]]), atol=1e-6)
    assert s.kept_per_token.tolist() == [[1, 2, 1]]


def test_cap_covering_the_union_is_lossless(usage):
    s = usage.cap_window_stats(_probs(_ROWS), top_k=2, cap=6)
    assert s.kept.tolist() == s.union.tolist()
    assert float(s.dropped_mass.abs().max()) < 1e-6
    assert s.kept_per_token.tolist() == [[2, 2, 2]]


def test_cap_below_tokens_is_refused(usage):
    # #721's union form needs T <= cap so every token's top-1 fits.
    with pytest.raises(ValueError, match="cap"):
        usage.cap_window_stats(_probs(_ROWS), top_k=2, cap=2 - 1)


def test_layers_are_independent(usage):
    two = torch.cat([_probs(_ROWS), _probs(_ROWS[::-1])])
    s = usage.cap_window_stats(two, top_k=2, cap=3)
    assert s.union.tolist() == [5, 5]
    assert s.dropped_mass.shape == (2, 3)


@pytest.mark.skipif(
    not hasattr(hf_common, "moe_union_pack"),
    reason="needs PR #721's hf_common.moe_union_pack",
)
@pytest.mark.parametrize("cap", [8, 16, 24])
def test_matches_pr721_moe_union_pack(usage, cap):
    torch.manual_seed(0)
    tokens, experts, top_k = 8, 128, 8
    probs = torch.softmax(torch.randn(tokens, experts) * 3, dim=-1)
    ids = torch.arange(experts, dtype=torch.float32)[:, None]
    slot_sticks, routes = hf_common.moe_union_pack(probs, top_k, cap, ids, 64)
    theirs = set(slot_sticks[:, 0].long().tolist())
    ours = set(
        usage.union_slot_mask(probs[None], top_k, cap)[0].nonzero().flatten().tolist()
    )
    assert ours == theirs
    s = usage.cap_window_stats(probs[None], top_k, cap)
    kept_mass = 1.0 - s.dropped_mass[0]
    # Their routes are renormalized over the kept slots: nonzero where we keep mass.
    assert torch.equal((routes > 0).sum(-1), s.kept_per_token[0])
    assert bool((kept_mass > 0).all())


def _record(prompt_len, rows_per_layer):
    probs = torch.cat([_probs(r) for r in rows_per_layer])  # [L, T, E]
    return {"prompt_len": prompt_len, "probs": probs}


def test_summarize_caps_windows_and_cost(usage):
    records = [_record(1, [_ROWS + [[1, 1, 1, 1, 1, 1]]])]  # 4 tokens, 3 generated
    model = usage.CostModel(fixed_ms=5.0, per_expert_ms=2.0)
    rows = usage.summarize_caps(
        records,
        windows=[3],
        caps=[3, 6],
        top_k=2,
        model=model,
        head_ms=10.0,
        decode_ms=20.0,
    )
    assert [(r.tokens, r.cap) for r in rows] == [(3, 3), (3, 6)]
    assert all(r.windows == 1 for r in rows)
    # The union form always reads ``cap`` slots, used or not.
    assert rows[0].verify_ms == pytest.approx(5.0 + 2.0 * 3 + 10.0)
    assert rows[1].verify_over_decode == pytest.approx((5.0 + 2.0 * 6 + 10.0) / 20.0)
    assert rows[1].dropped_mass_max == pytest.approx(0.0, abs=1e-6)
    assert rows[0].dropped_mass_max > 0


def test_summarize_caps_skips_caps_below_window(usage):
    records = [_record(0, [_ROWS])]
    model = usage.CostModel(fixed_ms=0.0, per_expert_ms=1.0)
    rows = usage.summarize_caps(
        records,
        windows=[3],
        caps=[2, 3],
        top_k=2,
        model=model,
        head_ms=0.0,
        decode_ms=1.0,
    )
    assert [r.cap for r in rows] == [3]


def test_cap_report_needs_probabilities(usage, tmp_path):
    routes = tmp_path / "routes.pt"
    torch.save(
        {"num_experts": 6, "top_k": 2, "records": [{"prompt_len": 0, "routes": None}]},
        routes,
    )
    args = usage.build_parser().parse_args(["cap-report", "--routes", str(routes)])
    with pytest.raises(SystemExit, match="--save-probs"):
        usage.run_cap_report(args)
