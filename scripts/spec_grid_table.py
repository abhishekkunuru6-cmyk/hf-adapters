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

"""One shape's cap x batch grid, per-request -> batched drafter, from bench JSONs.

Reads every row of the given ``bench_gemma4_batched_spec_spyre.py --output``
files and prints, per cap and batch size, end-to-end tok/s / plain tok/s at
the same B for the per-request drafter and for the batched one::

    | cap | B=1 | B=2 | ...
    | 16 | 2.42× -> 2.30× | ...

Only rows of the given shape with the given options (default: device argmax
and 8 draft threads) count. A cell is ``fail`` when it only ever failed and
``-`` when it never ran; when it ran ok more than once, the newest file wins.

Usage::

    python scripts/spec_grid_table.py --shape tree16 ~/xpress-al/overnight/*.json
    # does cap 32 B=2 with the batched drafter already have an ok row?
    python scripts/spec_grid_table.py --has 32 2 1 ~/xpress-al/overnight/*.json
"""

import argparse
import json
import os

DEFAULT_PLAIN = {1: 12.7, 2: 19.8, 4: 25.5, 8: 28.6}  # batched generate(), uncapped
BATCHED = "+bdraft"


def collect(paths, label):
    """``{(cap, batch, batched): row}`` for rows labelled ``label`` or
    ``label + BATCHED``. Ok rows beat failed ones; newer files beat older."""
    cells = {}
    for path in sorted(paths, key=os.path.getmtime):
        try:
            with open(path) as f:
                rows = json.load(f).get("rows", [])
        except (OSError, ValueError, AttributeError) as e:
            print(f"skipping {path}: {e}")
            continue
        for r in rows:
            if r.get("shape") not in (label, label + BATCHED):
                continue
            key = (str(r["cap"]), int(r["batch"]), r["shape"].endswith(BATCHED))
            old = cells.get(key)
            if old is None or r["status"] == "ok" or old["status"] != "ok":
                cells[key] = r
    return cells


def cell_text(row, plain):
    if row is None:
        return "-"
    if row["status"] != "ok":
        return "fail"
    return f"{row['tokens_per_s'] / plain:.2f}×"


def format_grid(cells, plain, caps=None, batches=None):
    caps = sorted(
        caps or {c for c, _, _ in cells},
        key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0),
    )
    batches = batches or sorted(plain)
    lines = [
        "| cap | " + " | ".join(f"B={b}" for b in batches) + " |",
        "|" + "---|" * (len(batches) + 1),
    ]
    for cap in caps:
        parts = []
        for b in batches:
            per, bat = cells.get((cap, b, False)), cells.get((cap, b, True))
            parts.append(f"{cell_text(per, plain[b])} -> {cell_text(bat, plain[b])}")
        lines.append(f"| {cap} | " + " | ".join(parts) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+", help="bench --output JSON files")
    p.add_argument("--shape", default="tree16")
    p.add_argument("--options", default="+devargmax+dt8", help="label suffix")
    p.add_argument(
        "--plain",
        nargs=4,
        type=float,
        metavar=("B1", "B2", "B4", "B8"),
        help="plain tok/s at B=1/2/4/8 (default: the measured baseline)",
    )
    p.add_argument(
        "--has",
        nargs=3,
        metavar=("CAP", "B", "BATCHED"),
        help="exit 0 if this cell (BATCHED: 0 or 1) has an ok row, else 1; no table",
    )
    args = p.parse_args()
    if args.plain and min(args.plain) <= 0:
        p.error("--plain values must be > 0")
    plain = dict(zip((1, 2, 4, 8), args.plain)) if args.plain else DEFAULT_PLAIN
    cells = collect(args.files, args.shape + args.options)
    if args.has:
        cap, batch, batched = args.has
        row = cells.get((cap, int(batch), batched == "1"))
        raise SystemExit(0 if row is not None and row["status"] == "ok" else 1)
    print(f"{args.shape}{args.options}: per-request -> batched drafter, x plain")
    print(format_grid(cells, plain))


if __name__ == "__main__":
    main()
