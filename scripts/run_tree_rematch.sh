#!/usr/bin/env bash
# tree16 rematch at batch (Gemma4 26B-A4B on #721), about 1.5-2 hours. Every
# part runs in its own process, because one compile failure makes every later
# compile in the same process fail. Each part writes $OUT/<name>.log (+ .json)
# and prints start / done lines here.
#
#   1. t16_c16_b8      tree16 cap 16 B=8 (128 rows), device argmax + 8 draft
#                      threads: is the overnight compile failure real?
#   2. t16_c64_b4      tree16 cap 64 B=4 (64 rows), same fixes: chain8 compiles
#                      64 rows at cap 64, and tree16 did in an earlier run
#   3. t16_c16_b2_bd   tree16 cap 16 B=2, same fixes + batched drafter
#   4. t16_c16_b4_bd   tree16 cap 16 B=4, same fixes + batched drafter: the fair
#                      rematch for chain8's 1.87x at B=4
#   5. t16_c16_b8_bd   tree16 cap 16 B=8, same fixes + batched drafter (only
#                      useful if part 1 compiles)
#   6. t16_c32_b1      tree16 cap 32 B=1 (16 rows), same fixes: the tree at a
#                      likely clean cap, vs chain8's 1.56x at B=1 cap 32
#   7. t16_c32_b2_bd   tree16 cap 32 B=2 (32 rows), same fixes + batched drafter
#
# Usage (Spyre pod, from #721's checkout; nothing outside $OUT is written):
#
#   cd ~/hf-adapters-721 && SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 nohup bash \
#       ~/xpress-al/batched/scripts/run_tree_rematch.sh > ~/xpress-al/rematch.log 2>&1 &
#
# Overrides: PY (python), OUT (output dir), N (draft threads, default 8), PARTS
# (names to run, default all, in the order above), PART_TIMEOUT (per part,
# default 40m). Environment variables such as CO_OPTIMIZING_LX_PLANNING pass
# through to every part unchanged.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/dt-inductor/.venv/bin/python}"
OUT="${OUT:-$HOME/xpress-al/overnight}"
N="${N:-8}"
ALL="t16_c16_b8 t16_c64_b4 t16_c16_b2_bd t16_c16_b4_bd t16_c16_b8_bd
t16_c32_b1 t16_c32_b2_bd"
PARTS="${PARTS:-$ALL}"
PART_TIMEOUT="${PART_TIMEOUT:-40m}"
LIMIT=()
if command -v timeout > /dev/null; then
    LIMIT=(timeout --signal=INT --kill-after=2m "$PART_TIMEOUT")
fi
mkdir -p "$OUT"

wanted() { # wanted <name>: is it in $PARTS?
    case " $(echo "$PARTS" | tr '\n' ' ') " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
    esac
}

bench() { # bench <name> <cap> <batch> [extra bench args...]
    local name="$1" cap="$2" batch="$3"
    shift 3
    wanted "$name" || return 0
    echo "$(date '+%F %T') start $name"
    ${LIMIT[@]+"${LIMIT[@]}"} "$PY" "$HERE/bench_gemma4_batched_spec_spyre.py" \
        --shapes tree16 --caps "$cap" --batch-sizes "$batch" \
        --device-argmax --draft-threads "$N" "$@" \
        --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    local rc=$? # before any other expansion resets $?
    local note=""
    [ "$rc" -eq 124 ] && note=" TIMED OUT after $PART_TIMEOUT"
    echo "$(date '+%F %T') done  $name (exit $rc$note) -> $OUT/$name.log"
}

echo "$(date '+%F %T') rematch start; out=$OUT, draft threads N=$N"
echo "host $(uname -n), #721 at $(git -C "$PWD" rev-parse --short HEAD 2> /dev/null || echo '?')"

bench t16_c16_b8 16 8
bench t16_c64_b4 64 4
bench t16_c16_b2_bd 16 2 --batched-drafter
bench t16_c16_b4_bd 16 4 --batched-drafter
bench t16_c16_b8_bd 16 8 --batched-drafter
bench t16_c32_b1 32 1
bench t16_c32_b2_bd 32 2 --batched-drafter

# One file with every table and the first error lines, for pasting.
{
    echo "# tree16 rematch $(date '+%F %T'), draft threads N=$N"
    for name in $PARTS; do
        f="$OUT/$name.log"
        [ -f "$f" ] || continue
        echo
        echo "## $name"
        grep '^| tree16' "$f" || echo "(no table)"
        grep -m3 -E 'DtException|Error|raised:' "$f"
    done
} > "$OUT/rematch_summary.md"
echo "$(date '+%F %T') rematch finished -> $OUT/rematch_summary.md"
