#!/usr/bin/env bash
# tree16's full cap x batch grid (Gemma4 26B-A4B on #721), per-request and
# batched drafter, the same table as chain8's. Caps 16 / 32 / 64, B 1 / 2 / 4
# / 8, device argmax + 8 draft threads: 24 cells. Every cell runs in its own
# process (a compile failure poisons later compiles in the same process). A
# cell is skipped when any JSON in $OUT already has an ok row for it (overnight,
# rematch, earlier grid runs) or when this grid already tried it (set RETRY=1
# to try failed cells again). Waits first until no other bench / probe /
# rematch is using the card. Ends by printing the table to
# $OUT/tree_grid.md. About 2.5-3 hours for the cells not run yet.
#
# Usage (Spyre pod, from #721's checkout; nothing outside $OUT is written):
#
#   cd ~/hf-adapters-721 && SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 nohup bash \
#       ~/xpress-al/batched/scripts/run_tree_grid.sh > ~/xpress-al/tree_grid.log 2>&1 &
#
# Overrides: PY, OUT, N (draft threads, default 8), CELLS (space-separated
# cap:batch:drafter, drafter pr or bd, default all, most useful first),
# PART_TIMEOUT (per cell, default 40m), RETRY.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/dt-inductor/.venv/bin/python}"
OUT="${OUT:-$HOME/xpress-al/overnight}"
N="${N:-8}"
RETRY="${RETRY:-0}"
PART_TIMEOUT="${PART_TIMEOUT:-40m}"
# cap 32 first (the likely clean cap at 16 rows), then cap 64, then the rest
ALL="32:1:pr 32:2:pr 32:2:bd 32:1:bd 32:4:pr 32:4:bd
64:1:bd 64:2:bd 64:4:pr 64:4:bd 64:8:pr 64:8:bd 32:8:pr 32:8:bd
16:1:pr 16:2:pr 16:4:pr 16:8:pr 16:1:bd 16:2:bd 16:4:bd 16:8:bd 64:1:pr 64:2:pr"
CELLS="${CELLS:-$ALL}"
TABLE=("$PY" "$HERE/spec_grid_table.py" --shape tree16 --options "+devargmax+dt$N")
LIMIT=()
if command -v timeout > /dev/null; then
    LIMIT=(timeout --signal=INT --kill-after=2m "$PART_TIMEOUT")
fi
mkdir -p "$OUT"

done_ok() { # done_ok <cap> <batch> <0|1>: an ok row in any JSON under $OUT?
    compgen -G "$OUT/*.json" > /dev/null || return 1
    "${TABLE[@]}" --has "$1" "$2" "$3" "$OUT"/*.json > /dev/null 2>&1
}

while pgrep -f 'bench_gemma4_batched_spec|probe_first_divergence|run_tree_rematch' > /dev/null; do
    sleep 60
done
echo "$(date '+%F %T') tree grid start; out=$OUT, draft threads N=$N"
echo "host $(uname -n), #721 at $(git -C "$PWD" rev-parse --short HEAD 2> /dev/null || echo '?')"

for cell in $CELLS; do
    IFS=: read -r cap batch drafter <<< "$cell"
    case "$drafter" in
    pr) extra=() bd=0 name="t16_c${cap}_b${batch}" ;;
    bd) extra=(--batched-drafter) bd=1 name="t16_c${cap}_b${batch}_bd" ;;
    *) echo "bad cell $cell (want cap:batch:pr|bd)" && continue ;;
    esac
    if done_ok "$cap" "$batch" "$bd"; then
        echo "$(date '+%F %T') skip  $name (already ok)"
        continue
    fi
    if [ "$RETRY" != 1 ] && [ -f "$OUT/$name.json" ]; then
        echo "$(date '+%F %T') skip  $name (tried before; RETRY=1 to rerun)"
        continue
    fi
    echo "$(date '+%F %T') start $name"
    ${LIMIT[@]+"${LIMIT[@]}"} "$PY" "$HERE/bench_gemma4_batched_spec_spyre.py" \
        --shapes tree16 --caps "$cap" --batch-sizes "$batch" \
        --device-argmax --draft-threads "$N" ${extra[@]+"${extra[@]}"} \
        --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    rc=$?
    note=""
    [ "$rc" -eq 124 ] && note=" TIMED OUT after $PART_TIMEOUT"
    echo "$(date '+%F %T') done  $name (exit $rc$note) -> $OUT/$name.log"
done

"${TABLE[@]}" "$OUT"/*.json > "$OUT/tree_grid.md" 2>&1
cat "$OUT/tree_grid.md"
echo "$(date '+%F %T') tree grid finished -> $OUT/tree_grid.md"
