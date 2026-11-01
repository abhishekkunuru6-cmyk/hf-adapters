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


"""CPU tests for ``scripts/probe_first_divergence_spyre.py`` (why speculative
outputs first differ from plain greedy)."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("hf_adapters._xpress_tree")
helpers = pytest.importorskip("tests._gemma4_tree_helpers")

_SCRIPTS = Path(__file__).parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe_mod():
    return _load("probe_first_divergence_spyre")


def test_divergence_row(probe_mod):
    logits = torch.tensor([0.0, 5.0, 4.98, 1.0, 4.0])
    r = probe_mod.divergence_row(logits, ref_token=1, spec_token=2, k=3)
    assert r["margin"] == pytest.approx(0.02, abs=1e-4)
    assert (r["ref_rank"], r["spec_rank"]) == (1, 2)
    assert r["prefill_picks"] == "plain" and r["finite"]
    assert r["top1_gap"] == pytest.approx(0.02, abs=1e-4)
    assert [t for t, _ in r["top"]] == [1, 2, 4]
    r = probe_mod.divergence_row(logits, ref_token=4, spec_token=1)
    assert r["prefill_picks"] == "spec" and r["margin"] < 0 and r["ref_rank"] == 3
    r = probe_mod.divergence_row(logits, ref_token=3, spec_token=4)
    assert r["prefill_picks"] == "other"


def test_probe_on_a_tiny_model(probe_mod, monkeypatch):
    """Equal outputs give no divergence; a forced one is probed at its position."""
    from tests.cpu.test_gemma4_spec_decode import _drafter

    from hf_adapters import _xpress_tree

    reference, adapted = helpers.tiny_gemma4_moe(monkeypatch)
    helpers.run_eagerly(monkeypatch)
    helpers._unwrap(adapted)
    monkeypatch.setattr(_xpress_tree.torch, "compile", lambda fn, **_: fn)
    bsd = _load("batched_spec_decode")
    bench = _load("bench_gemma4_batched_spec_spyre")
    bsd.configure_union_cap(adapted, None)
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(2, 320, (n,), generator=g) for n in (21, 30)]
    refs = [
        reference.generate(
            p[None], max_new_tokens=10, do_sample=False, min_new_tokens=10
        )[0, p.numel() :].tolist()
        for p in prompts
    ]
    decoder = bsd.BatchedSpecDecoder(
        adapted,
        _drafter(),
        bsd.SHAPES["chain4"],
        drafter_kwargs=dict(dtype=torch.float32, device="cpu", compile=False),
    )
    rows = probe_mod.probe(decoder, prompts, refs, 10, 320, bench.first_difference)
    assert [r["first_diff"] for r in rows] == [None, None]  # exact greedy on CPU

    real = decoder.generate

    def flipped(batch, n):
        out = real(batch, n)
        return [o[:4] + [(o[4] + 1) % 320] + o[5:] for o in out]

    monkeypatch.setattr(decoder, "generate", flipped)
    rows = probe_mod.probe(decoder, prompts, refs, 10, 320, bench.first_difference)
    for r, ref in zip(rows, refs):
        assert r["first_diff"] == 4 and r["ref_token"] == ref[4]
        assert r["prefill_picks"] == "plain"  # the prefill agrees with greedy
        assert r["margin"] > 0 and r["ref_rank"] == 1 and r["finite"]
    table = probe_mod.format_rows("chain4", rows)
    assert table.count("\n") == 3 and "| chain4 | 1 | 0 | 4 |" in table

    # batch 2: both prompts decoded together, each probed on its own prefix
    calls = []

    def batched(batch, n):
        calls.append(len(batch))
        return flipped(batch, n)

    monkeypatch.setattr(decoder, "generate", batched)
    rows2 = probe_mod.probe(
        decoder, prompts, refs, 10, 320, bench.first_difference, batch=2
    )
    assert calls == [2]
    assert [r["batch"] for r in rows2] == [2, 2]
    assert [r["ref_token"] for r in rows2] == [r["ref_token"] for r in rows]
    assert [r["margin"] for r in rows2] == pytest.approx([r["margin"] for r in rows])
    with pytest.raises(ValueError):
        probe_mod.probe(decoder, prompts, refs, 10, 320, bench.first_difference, 3)
