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

"""CPU tests for the pure helpers of ``scripts/measure_moe_expert_usage.py``."""

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parents[2] / "scripts" / "measure_moe_expert_usage.py"


@pytest.fixture(scope="module")
def usage():
    spec = importlib.util.spec_from_file_location("measure_moe_expert_usage", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _routes():
    # 2 layers x 4 tokens x top-2, 8 experts.
    return torch.tensor(
        [
            [[0, 1], [1, 2], [0, 1], [5, 6]],  # layer 0
            [[3, 4], [3, 4], [3, 4], [3, 4]],  # layer 1: every token agrees
        ]
    )


def test_distinct_per_layer(usage):
    routes = _routes()
    assert usage.distinct_per_layer(routes, 0, 1).tolist() == [2, 2]
    assert usage.distinct_per_layer(routes, 0, 2).tolist() == [3, 2]
    assert usage.distinct_per_layer(routes, 0, 4).tolist() == [5, 2]
    assert usage.distinct_per_layer(routes, 2, 2).tolist() == [4, 2]
    with pytest.raises(ValueError, match="window"):
        usage.distinct_per_layer(routes, 3, 2)


def test_window_means_slide_over_generated_tokens(usage):
    record = {"prompt_len": 1, "routes": _routes()}
    # Windows of 2 starting at 1 and 2: layer means (3+2)/2=2.5 and (4+2)/2=3.
    assert usage.window_means([record], 2) == pytest.approx([2.5, 3.0])
    # A window longer than the generated part yields nothing.
    assert usage.window_means([record], 4) == []


def test_random_distinct(usage):
    assert usage.random_distinct(128, 8, 1) == pytest.approx(8.0)
    assert usage.random_distinct(128, 8, 16) == pytest.approx(82.4, abs=0.1)
    assert usage.random_distinct(128, 8, 1000) == pytest.approx(128.0)


def test_cost_model_from_two_points(usage):
    model = usage.CostModel.from_points(
        decode_backbone_ms=64.0,
        decode_experts=8,
        all_backbone_ms=954.0,
        all_experts=128,
    )
    assert model.per_expert_ms == pytest.approx(890.0 / 120)
    assert model.backbone_ms(8) == pytest.approx(64.0)
    assert model.backbone_ms(128) == pytest.approx(954.0)
    with pytest.raises(ValueError, match="distinct"):
        usage.CostModel.from_points(
            decode_backbone_ms=64.0,
            decode_experts=8,
            all_backbone_ms=954.0,
            all_experts=8,
        )


def test_verify_over_decode(usage):
    model = usage.CostModel(fixed_ms=4.0, per_expert_ms=7.5)
    # backbone(8)=64, + verify head 14 -> 78 over a 78 ms decode = 1.0
    assert usage.verify_over_decode(model, 8, head_ms=14.0, decode_ms=78.0) == (
        pytest.approx(1.0)
    )


def test_percentile(usage):
    assert usage.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert usage.percentile([5.0], 0.9) == 5.0
    with pytest.raises(ValueError):
        usage.percentile([], 0.9)


def test_summarize_rows_and_json(usage):
    record = {"prompt_len": 0, "routes": _routes()}
    model = usage.CostModel(fixed_ms=4.0, per_expert_ms=7.5)
    rows = usage.summarize_windows(
        [record],
        [1, 2],
        num_experts=8,
        top_k=2,
        model=model,
        head_ms=14.0,
        decode_ms=78.0,
    )
    assert [r.tokens for r in rows] == [1, 2]
    assert rows[0].mean_experts == pytest.approx(2.0)
    assert rows[0].windows == 4
    ratios = usage.ratios_json(rows)
    assert set(ratios["verify_over_decode"]) == {"1", "2"}
    assert ratios["verify_over_decode"]["1"] == pytest.approx(
        rows[0].verify_over_decode
    )


def test_window_counts(usage):
    record = {"prompt_len": 0, "routes": _routes()}
    counts = usage.window_counts([record], 2)
    # Windows start at 0, 1, 2; layer 0 sees {0,1,2}, {0,1,2}, {0,1,5,6}.
    assert counts.tolist() == [[3, 2], [3, 2], [4, 2]]
    assert usage.window_counts([record], 5).shape == (0, 2)


def test_worst_case_slots(usage):
    """A static-shape kernel must load the busiest window's experts: per layer
    (each layer sized to its own max) or one size for all layers."""
    record = {"prompt_len": 0, "routes": _routes()}
    model = usage.CostModel(fixed_ms=4.0, per_expert_ms=7.5)
    (row,) = usage.summarize_windows(
        [record],
        [2],
        num_experts=8,
        top_k=2,
        model=model,
        head_ms=14.0,
        decode_ms=78.0,
    )
    assert row.mean_experts == pytest.approx((2.5 + 2.5 + 3.0) / 3)
    assert row.per_layer_slots == pytest.approx((4 + 2) / 2)
    assert row.uniform_slots == 4
    assert row.static_verify_over_decode == pytest.approx(
        (model.backbone_ms(3.0) + 14.0) / 78.0
    )
    assert row.uniform_verify_over_decode == pytest.approx(
        (model.backbone_ms(4) + 14.0) / 78.0
    )
    ratios = usage.ratios_json([row])
    assert ratios["verify_over_decode_static"]["2"] == pytest.approx(
        row.static_verify_over_decode
    )
    assert ratios["verify_over_decode_uniform"]["2"] == pytest.approx(
        row.uniform_verify_over_decode
    )


def test_truncate_routes_keeps_the_highest_scoring_experts(usage):
    record = {"prompt_len": 1, "routes": _routes()}
    (cut,) = usage.truncate_routes([record], 1)
    assert cut["routes"].tolist() == [[[0], [1], [0], [5]], [[3], [3], [3], [3]]]
    assert cut["prompt_len"] == 1
    assert record["routes"].shape[-1] == 2  # input left untouched
    assert usage.truncate_routes([record], 2)[0]["routes"].shape[-1] == 2
    for bad in (0, 3):
        with pytest.raises(ValueError, match="top_k"):
            usage.truncate_routes([record], bad)


def test_decode_ms_for_top_k(usage):
    model = usage.CostModel(fixed_ms=5.0, per_expert_ms=7.0)
    # Halving top-8 to top-4 saves 4 expert loads per layer at decode.
    assert usage.decode_ms_for_top_k(77.0, model, 8, 4) == pytest.approx(49.0)
    assert usage.decode_ms_for_top_k(77.0, model, 8, 8) == pytest.approx(77.0)
