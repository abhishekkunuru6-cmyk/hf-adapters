#!/usr/bin/env bash
# Overnight data collection for chain vs tree on Spyre (Gemma4 26B-A4B on #721),
# about 6 hours. Parts run one after another, each in its own process, most
# important first: a failing or crashing part loses only itself. Each part
# writes $OUT/<name>.log (+ .json) and prints start / done lines here.
#
#   1. draft_threads    chain8 draft profile at 0 (default) 8 4 2 1 threads; the
#                       fastest count becomes N for every later part (A1)
#   2. chain8_cap64     chain8 B 1 2 4 8, cap 64, device argmax + N draft threads
#   3. chain8_cap16     chain8 B 2 4 8 1, cap 16, device argmax + N draft threads
#                       (B=1 last: round 1's failure came at B=2 right after B=1)
#   4. tree16_cap64     the tree rematch with the same fixes and columns
#   5. tree16_cap16
#   6. probe            token-17 question: first divergence from plain greedy at
#                       cap off, chain8 + tree16, with the target's logits there
#   7. chain8_cap64_dt  B 1 8, N draft threads only (splits A1 from A8)
#   8. chain16_cap16    B 1 2 4, N draft threads (chain16 lost on draft time)
#   9. capoff           chain8 B 1 4, cap off, N draft threads: exact baseline
#  10. devargmax_cap16  chain8 B 1 then 2, cap 16, device argmax: round 1's order,
#                       to reproduce its failure with the full error
#
# Grid parts 2-9 use --compare-ar (match_ar = quality vs plain greedy).
#
# Usage (Spyre pod, from #721's checkout; nothing outside $OUT is written):
#
#   cd ~/hf-adapters-721 && SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 nohup bash \
#       ~/xpress-al/batched/scripts/run_overnight.sh > ~/xpress-al/overnight.log 2>&1 &
#
# Overrides: PY (python), OUT (output dir), N (skip the pick), PARTS (names to
# run, default all, in the order above), PART_TIMEOUT (per part, default 100m:
# a hung part is stopped so the rest still run; rows already done are kept,
# since the bench rewrites its JSON after every row).

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/dt-inductor/.venv/bin/python}"
OUT="${OUT:-$HOME/xpress-al/overnight}"
ALL="draft_threads chain8_cap64 chain8_cap16 tree16_cap64 tree16_cap16 probe
chain8_cap64_dt chain16_cap16 capoff devargmax_cap16"
PARTS="${PARTS:-$ALL}"
FALLBACK_THREADS=8 # the breakdown's 4-8 were as fast as 72 in a hot loop
PART_TIMEOUT="${PART_TIMEOUT:-100m}"
LIMIT=()
if command -v timeout > /dev/null; then
    LIMIT=(timeout --signal=INT --kill-after=2m "$PART_TIMEOUT")
fi
mkdir -p "$OUT"

part() { # part <name> <command...>
    local name="$1"
    shift
    echo "$(date '+%F %T') start $name"
    "${LIMIT[@]}" "$@" > "$OUT/$name.log" 2>&1
    local rc=$? # before any other expansion resets $?
    local note=""
    [ "$rc" -eq 124 ] && note=" TIMED OUT after $PART_TIMEOUT"
    echo "$(date '+%F %T') done  $name (exit $rc$note) -> $OUT/$name.log"
}

wanted() { # wanted <name>: is it in $PARTS?
    case " $(echo "$PARTS" | tr '\n' ' ') " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
    esac
}

bench() { # bench <name> <bench args...>
    local name="$1"
    shift
    wanted "$name" || return 0
    part "$name" "$PY" "$HERE/bench_gemma4_batched_spec_spyre.py" \
        --output "$OUT/$name.json" "$@"
}

echo "$(date '+%F %T') overnight start; out=$OUT"
echo "host $(uname -n), $(nproc 2> /dev/null || getconf _NPROCESSORS_ONLN) cpus, #721 at $(git -C "$PWD" rev-parse --short HEAD 2> /dev/null || echo '?')"

if wanted draft_threads; then
    part draft_threads "$PY" "$HERE/profile_xpress_draft_spyre.py" \
        --shapes chain8 --threads 0 8 4 2 1 --output "$OUT/draft_threads.json"
fi
if [ -z "${N:-}" ]; then
    N="$("$PY" "$HERE/profile_xpress_draft_spyre.py" --shapes chain8 \
        --pick-threads "$OUT/draft_threads.json" 2> /dev/null)" || N=""
    case "$N" in
    '' | *[!0-9]*) N="$FALLBACK_THREADS" && echo "thread pick failed; using $N" ;;
    esac
fi
echo "$(date '+%F %T') draft threads N=$N"

bench chain8_cap64 --shapes chain8 --caps 64 --batch-sizes 1 2 4 8 \
    --device-argmax --draft-threads "$N" --compare-ar
bench chain8_cap16 --shapes chain8 --caps 16 --batch-sizes 2 4 8 1 \
    --device-argmax --draft-threads "$N" --compare-ar
bench tree16_cap64 --shapes tree16 --caps 64 --batch-sizes 1 2 4 8 \
    --device-argmax --draft-threads "$N" --compare-ar
bench tree16_cap16 --shapes tree16 --caps 16 --batch-sizes 2 4 8 1 \
    --device-argmax --draft-threads "$N" --compare-ar

if wanted probe; then
    part probe "$PY" "$HERE/probe_first_divergence_spyre.py" \
        --shapes chain8 tree16 --cap off --output "$OUT/probe.json"
fi

bench chain8_cap64_dt --shapes chain8 --caps 64 --batch-sizes 1 8 \
    --draft-threads "$N" --compare-ar
bench chain16_cap16 --shapes chain16 --caps 16 --batch-sizes 1 2 4 \
    --draft-threads "$N" --compare-ar
bench capoff --shapes chain8 --caps off --batch-sizes 1 4 \
    --draft-threads "$N" --compare-ar
bench devargmax_cap16 --shapes chain8 --caps 16 --batch-sizes 1 2 --device-argmax

# One file with every table, for pasting.
{
    echo "# overnight $(date '+%F %T'), draft threads N=$N"
    for name in $PARTS; do
        f="$OUT/$name.log"
        [ -f "$f" ] || continue
        echo
        echo "## $name"
        grep '^|' "$f" || echo "(no table)"
        grep -A3 "raised:" "$f" | head -8
    done
} > "$OUT/summary.md"
echo "$(date '+%F %T') overnight finished -> $OUT/summary.md"
