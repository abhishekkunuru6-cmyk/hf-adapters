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

"""CPU tests for ``scripts/spec_grid_table.py`` (one shape's cap x batch grid)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "spec_grid_table.py"
LABEL = "tree16+devargmax+dt8"


@pytest.fixture(scope="module")
def grid():
    spec = importlib.util.spec_from_file_location("spec_grid_table", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(shape, cap, batch, tok_s=None):
    status = "ok" if tok_s is not None else "fail"
    return dict(
        shape=shape, cap=cap, batch=batch, status=status, tokens_per_s=tok_s or 0
    )


def _write(path, rows, age):
    path.write_text(json.dumps(dict(env={}, rows=rows)))
    os.utime(path, (age, age))
    return str(path)


def test_collect_and_format(grid, tmp_path):
    old = _write(
        tmp_path / "old.json",
        [
            _row(LABEL, "16", 1, 25.4),  # superseded by the newer ok row
            _row(LABEL, "16", 8),  # fail only
            _row(LABEL, "64", 4, 30.0),  # ok, later failed: ok wins
            _row("chain8+devargmax+dt8", "16", 1, 99.0),  # other shape
            _row("tree16", "16", 2, 99.0),  # other options
        ],
        1000,
    )
    new = _write(
        tmp_path / "new.json",
        [
            _row(LABEL, "16", 1, 30.7),
            _row(LABEL + "+bdraft", "16", 1, 25.4),
            _row(LABEL, "64", 4),
        ],
        2000,
    )
    cells = grid.collect([new, old], LABEL)
    assert cells[("16", 1, False)]["tokens_per_s"] == 30.7
    assert cells[("64", 4, False)]["status"] == "ok"
    assert ("16", 2, False) not in cells
    table = grid.format_grid(cells, grid.DEFAULT_PLAIN, batches=[1, 4, 8])
    lines = table.splitlines()
    assert lines[0] == "| cap | B=1 | B=4 | B=8 |"
    assert lines[2] == "| 16 | 2.42× -> 2.00× | - -> - | fail -> - |"
    assert lines[3] == "| 64 | - -> - | 1.18× -> - | - -> - |"


def test_bad_file_is_skipped(grid, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert grid.collect([str(bad)], LABEL) == {}
    assert "skipping" in capsys.readouterr().out


def test_has_exit_code(tmp_path):
    f = _write(tmp_path / "a.json", [_row(LABEL + "+bdraft", "32", 2, 30.0)], 1000)

    def has(*cell):
        cmd = [sys.executable, str(_SCRIPT), "--has", *cell, f]
        return subprocess.run(cmd, capture_output=True).returncode

    assert has("32", "2", "1") == 0
    assert has("32", "2", "0") == 1
    assert has("16", "1", "1") == 1
