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

"""CPU tests for the pure helpers of ``scripts/spike_verify_cost_spyre.py``."""

import importlib.util
from pathlib import Path

import pytest

from hf_adapters import hf_common

_SCRIPT = Path(__file__).parents[2] / "scripts" / "spike_verify_cost_spyre.py"


@pytest.fixture(scope="module")
def spike():
    spec = importlib.util.spec_from_file_location("spike_verify_cost", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(spike, **overrides):
    values = dict(
        decode_backbone_ms=40.0,
        verify_backbone_ms=90.0,
        decode_head_ms=10.0,
        verify_head_ms=10.0,
        draft_ms=50.0,
        verify_rows=64,
        block_tokens=16,
    )
    return spike.SpikeResult(**{**values, **overrides})


def test_cost_model(spike):
    result = _result(spike)
    assert result.decode_ms == 50.0
    assert result.verify_ms == 100.0
    assert result.verify_over_decode == 2.0
    # (verify 100 + draft 50) / decode 50: a block must yield 3 tokens to tie.
    assert result.break_even_tokens == 3.0
    assert result.speedup_at(3) == pytest.approx(1.0)
    assert result.speedup_at(6) == pytest.approx(2.0)


def test_split_drops_first_chunk_and_keeps_decode(spike):
    records = [(64, 900.0), (64, 80.0), (64, 82.0), (1, 40.0), (1, 41.0)]
    decode, verify = spike.split_backbone_samples(records, verify_rows=64)
    assert decode == [40.0, 41.0]
    assert verify == [80.0, 82.0]


def test_median_requires_samples(spike):
    assert spike.median_ms([3.0, 1.0, 2.0], label="x") == 2.0
    with pytest.raises(RuntimeError, match="no verify samples"):
        spike.median_ms([], label="verify")


@pytest.mark.parametrize(
    "prompt, rows, block, steps, match",
    [
        (448, 16, 16, 32, "verify-rows must be a positive multiple of 64"),
        (448, 64, 1, 32, "block-tokens must be in"),
        (448, 64, 65, 32, "block-tokens must be in"),
        (448, 64, 16, 0, "decode-steps must be >= 1"),
        (480, 64, 16, 32, "multiple of --verify-rows"),
        (64, 64, 16, 32, "at least 2 prefill chunks"),
    ],
)
def test_check_geometry_rejects(spike, prompt, rows, block, steps, match):
    with pytest.raises(ValueError, match=match):
        spike.check_geometry(prompt, rows, block, steps)


def test_check_geometry_accepts_defaults(spike):
    spike.check_geometry(
        spike.DEFAULT_PROMPT_TOKENS,
        spike.DEFAULT_VERIFY_ROWS,
        spike.DEFAULT_BLOCK_TOKENS,
        spike.DEFAULT_DECODE_STEPS,
    )


@pytest.mark.parametrize(
    "prompt, new_tokens", [(448, 33), (480, 33), (512, 33), (1000, 64)]
)
def test_key_widths_match_generate(spike, prompt, new_tokens):
    """The script's Lk arithmetic must track hf_common.generate's sizing."""
    assert spike.key_widths(prompt, new_tokens) == (
        hf_common._sdpa_compatible_kv_length(prompt),
        hf_common.generation_cache_len(prompt, new_tokens),
    )


def test_default_geometry_attends_over_equal_widths(spike):
    prefill_lk, decode_lk = spike.key_widths(
        spike.DEFAULT_PROMPT_TOKENS, spike.DEFAULT_DECODE_STEPS + 1
    )
    assert prefill_lk == decode_lk == 512


def test_report_mentions_break_even(spike):
    text = spike.format_report(_result(spike), samples=(32, 27))
    assert "break-even tokens/block    : 3.00" in text
    assert "verify / decode            : 2.00x" in text
    assert "verify, 64 rows" in text
    assert "a block yields at most 16" in text


def test_check_geometry_sub_stick_rows(spike):
    spike.check_geometry(448, 16, 16, 32, row_multiple=16)
    with pytest.raises(ValueError, match="multiple of 64"):
        spike.check_geometry(448, 16, 16, 32)
    with pytest.raises(ValueError, match="--row-multiple must be one of"):
        spike.check_geometry(448, 24, 16, 32, row_multiple=24)


def test_key_widths_follow_row_multiple(spike):
    # 448 + round16(33) = 496 -> Lk 512, the same width as the 64-row default.
    assert spike.key_widths(448, 33, row_multiple=16) == (512, 512)


def test_summarize_stages_averages_per_forward(spike):
    records = [
        ("attn", 64, 2.0),
        ("ffn", 64, 10.0),
        ("attn", 64, 4.0),
        ("ffn", 64, 14.0),
        ("decode", 1, 3.0),
    ]
    stages = spike.summarize_stages(records, {64: 2, 1: 1})
    assert stages == {("attn", 64): 3.0, ("ffn", 64): 12.0, ("decode", 1): 3.0}
    assert "ffn rows=64" in spike.format_stages(stages)


def test_summarize_stages_skips_unknown_widths(spike):
    assert spike.summarize_stages([("attn", 16, 1.0)], {64: 1}) == {}


def _fake_adapter():
    """Mimics hf_qwen3: prefill uses ``_run_backbone_forward``; decode goes
    through ``_run_forward``, which calls the same function by another global
    name (``standard_gqa_backbone_forward`` in hf_common)."""
    import types

    import torch

    module = types.ModuleType("fake_adapter")
    exec(
        "def backbone(model, input_ids, *args, **kwargs):\n"
        "    return input_ids.float()\n"
        "def _run_forward(model, input_ids):\n"
        "    return backbone(model, input_ids) + 1\n"
        "_run_backbone_forward = backbone\n",
        module.__dict__,
    )
    return module, torch


def test_backbone_timer_catches_prefill_and_decode(spike):
    module, torch = _fake_adapter()
    timer = spike.install_backbone_timer(module, sync=lambda: None)

    module._run_backbone_forward(None, torch.zeros(1, 64))  # prefill path
    out = module._run_forward(None, torch.zeros(1, 1))  # decode path
    assert [seq for seq, _ in timer.records] == [64, 1]
    assert out.item() == 1.0  # the wrapped function still runs


def test_moe_only_options_need_the_gemma4_moe_adapter(spike):
    spike.check_adapter_options("hf_adapters.hf_gemma4_moe", "gather", True)
    spike.check_adapter_options("hf_adapters.hf_qwen3", "prefill", False)
    with pytest.raises(ValueError, match="--moe-path gather"):
        spike.check_adapter_options("hf_adapters.hf_qwen3", "gather", False)
    with pytest.raises(ValueError, match="--split-timing"):
        spike.check_adapter_options("hf_adapters.hf_qwen3", "prefill", True)
