#!/usr/bin/env bash
# Chain optimization round 1 on Spyre (Gemma4 26B-A4B on #721), one part after
# another; a failing part does not stop the next. Each part writes its own log
# (and JSON) under $OUT, then prints a "done" line here.
#
#   1. draft profile: forward vs host search, chain8 / tree16 / chain8
#   2. grid, one process per shape in $SHAPES (a native crash loses only that
#      shape): default chain8 x B 1 2 4 8 x caps 16 64, with the readback /
#      tok/row / active / slow / compiles columns. SHAPES="chain8 chain8-beam
#      chain4 chain6" adds the parked acceptance variants.
#   3. the chain8 rows again with --device-argmax (a new device reduction)
#
# Every bench part rewrites its JSON after each row, so finished rows survive
# a crash.
#
# Usage (Spyre pod, from #721's checkout; nothing outside $OUT is written):
#
#   SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 nohup bash \
#       ~/xpress-al/batched/scripts/run_chain_round1.sh > round1.log 2>&1 &
#
# Overrides: PY (python), OUT (output dir), SHAPES, BATCHES, CAPS.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/dt-inductor/.venv/bin/python}"
OUT="${OUT:-$HOME/xpress-al/round1}"
BATCHES="${BATCHES:-1 2 4 8}"
CAPS="${CAPS:-16 64}"
SHAPES="${SHAPES:-chain8}"
mkdir -p "$OUT"

part() {  # part <name> <command...>
    local name="$1"
    shift
    echo "$(date '+%F %T') start $name"
    "$@" > "$OUT/$name.log" 2>&1
    local rc=$?  # before any other expansion resets $?
    echo "$(date '+%F %T') done  $name (exit $rc) -> $OUT/$name.log"
}

part draft_profile "$PY" "$HERE/profile_xpress_draft_spyre.py" \
    --shapes chain8 tree16 chain8

# shellcheck disable=SC2086  # SHAPES / BATCHES / CAPS are word lists
for shape in $SHAPES; do
    part "grid_$shape" "$PY" "$HERE/bench_gemma4_batched_spec_spyre.py" \
        --shapes "$shape" \
        --batch-sizes $BATCHES --caps $CAPS --output "$OUT/grid_$shape.json"
done

# shellcheck disable=SC2086
part devargmax "$PY" "$HERE/bench_gemma4_batched_spec_spyre.py" \
    --shapes chain8 --device-argmax \
    --batch-sizes $BATCHES --caps $CAPS --output "$OUT/devargmax.json"

echo "$(date '+%F %T') round 1 finished"
