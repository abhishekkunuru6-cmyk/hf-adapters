#!/bin/bash
# Re-measure the XPress / Gemma4 MoE verify numbers on the current build.
#
#   1. Spyre spikes (one at a time): Gemma4 64-row and 16-row verify, Qwen3-8B 16-row.
#   2. Expert-usage report (CPU) with the new spike costs, top-8 and top-4.
#   3. Acceptance replays (CPU, in parallel): K 0/6 x ctx 64/1024 x verify 4/6/8/16 under
#      today's all-experts kernel, N variable, N fixed per layer, N fixed uniform.
#
# Usage: nohup scripts/run_xpress_moe_eval.sh > ~/xpress-al/new/run_all.log 2>&1 &
# Env overrides: OUT, TAPS, ROUTES, DRAFT_MS.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/xpress-al/new}
TAPS=${TAPS:-$HOME/xpress-al/taps.pt}
ROUTES=${ROUTES:-$HOME/xpress-al/routes.pt}
DRAFT_MS=${DRAFT_MS:-50.4}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

{
  git log --oneline -1
  python -c "import torch, torch_spyre; print('torch', torch.__version__, '| torch_spyre', torch_spyre.__file__)"
} > "$OUT/versions.txt" 2>&1

log "spike gemma4 r64"
python scripts/spike_verify_cost_spyre.py > "$OUT/spike_gemma4_r64.log" 2>&1
log "spike gemma4 r16"
python scripts/spike_verify_cost_spyre.py --verify-rows 16 --row-multiple 16 \
  > "$OUT/spike_gemma4_r16.log" 2>&1
log "spike qwen3 r16"
python scripts/spike_verify_cost_spyre.py --model-path Qwen/Qwen3-8B \
  --verify-rows 16 --row-multiple 16 > "$OUT/spike_qwen3_r16.log" 2>&1

# Cost-model inputs from the 16-row Gemma4 spike; the 64-row one if 16 rows failed.
SRC="$OUT/spike_gemma4_r16.log"
if ! grep -q "verify, 16 rows" "$SRC"; then
  log "r16 spike has no result -> using the r64 numbers"
  SRC="$OUT/spike_gemma4_r64.log"
fi
read -r DEC_BB _ DEC_MS < <(grep "decode, 1 token" "$SRC" | tail -n 1 | awk '{print $5, $6, $7}')
read -r VER_BB VER_HEAD _ < <(grep -E "verify, [0-9]+ rows" "$SRC" | tail -n 1 | awk '{print $5, $6, $7}')
TODAY=$(grep "verify / decode" "$SRC" | tail -n 1 | sed -E 's/.*: *([0-9.]+)x.*/\1/')
echo "source=$SRC DEC_BB=$DEC_BB DEC_MS=$DEC_MS VER_BB=$VER_BB VER_HEAD=$VER_HEAD TODAY=$TODAY" \
  | tee "$OUT/cost_inputs.txt"
if [ -z "$DEC_BB" ] || [ -z "$VER_BB" ] || [ -z "$TODAY" ]; then
  log "could not parse the spike summary; stopping before the CPU stages"
  exit 1
fi

for K in 8 4; do
  log "report top-$K"
  python scripts/measure_moe_expert_usage.py report --routes "$ROUTES" --top-k "$K" \
    --decode-backbone-ms "$DEC_BB" --all-backbone-ms "$VER_BB" \
    --verify-head-ms "$VER_HEAD" --decode-ms "$DEC_MS" \
    --out-json "$OUT/verify_cost_top$K.json" > "$OUT/expert_usage_top$K.log" 2>&1
done

export OMP_NUM_THREADS=$(( $(nproc) / 4 ))
COMMON=(--taps "$TAPS" --passes 0 6 --predecessor real --ctx-pad 64 1024
        --verify-tokens 4 6 8 16 --draft-ms "$DRAFT_MS" --decode-ms "$DEC_MS")
log "replays (4 in parallel, OMP_NUM_THREADS=$OMP_NUM_THREADS)"
python scripts/measure_xpress_acceptance.py replay "${COMMON[@]}" \
  --verify-over-decode "$TODAY" > "$OUT/replay_today.log" 2>&1 &
for KEY in verify_over_decode verify_over_decode_static verify_over_decode_uniform; do
  python scripts/measure_xpress_acceptance.py replay "${COMMON[@]}" \
    --verify-cost-json "$OUT/verify_cost_top8.json" --verify-cost-key "$KEY" \
    > "$OUT/replay_$KEY.log" 2>&1 &
done
wait
log "ALL DONE"
