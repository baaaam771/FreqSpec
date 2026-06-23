#!/usr/bin/env bash
# run_sr_coupling_experiments.sh — SR reviewer-proofing experiments.
#
#   (A) coupling x tolerance operating-point sweep   (coupling 0/0.3/0.6/0.9)
#   (B) matched-cost reduced-step targets            (Target-44/45/48 vs FreqSpec NFE)
#   -> assembled coupling_table.{csv,tex} with AURC merged per coupling.
#
# The coupling=0 run also produces the matched-cost target baselines (target
# steps 50 48 45 44 40 30), so FreqSpec-default (NFE ~43.7) sits between
# Target-44 and Target-45, and FreqSpec-strict (NFE ~47.6) next to Target-48.
#
# Usage:
#   ./run_sr_coupling_experiments.sh <data_root> <draft_ckpt> <out_root> [num_images]
set -euo pipefail

DATA_ROOT="$1"
DRAFT_CKPT="$2"
OUT_ROOT="$3"
N="${4:-100}"
TARGET_ID="stabilityai/stable-diffusion-x4-upscaler"

COMMON=(--target_id "$TARGET_ID" --draft_ckpt "$DRAFT_CKPT" --use_ema_draft
        --data_root "$DATA_ROOT" --num_images "$N" --lr_size 128 --scale 4
        --blend_temperature 0.10 --x0_thr_strict 0.02 --x0_thr_loose 0.07
        --x0_strict_center 0.45 --x0_strict_width 0.12 --drift_k_switch_threshold 0.006)

# (A0 + B) coupling=0: full targets (incl. matched-cost) + freqspec
echo "=== coupling=0.0 (with matched-cost targets 50 48 45 44 40 30) ==="
python sr_baseline_sweep.py "${COMMON[@]}" \
  --out_root "$OUT_ROOT/cpl_0.0" --target_steps 50 48 45 44 40 30 \
  --saliency_x0_coupling 0.0
python analyze_sr.py --out_root "$OUT_ROOT/cpl_0.0" --ref_method target_s50

# (A) coupling = 0.3 / 0.6 / 0.9: freqspec only (targets already covered)
for C in 0.3 0.6 0.9; do
  echo "=== coupling=$C (freqspec only) ==="
  python sr_baseline_sweep.py "${COMMON[@]}" \
    --out_root "$OUT_ROOT/cpl_$C" --freqspec_only \
    --saliency_x0_coupling "$C"
  # reuse coupling=0 target_s50 as the LPIPSt reference
  cp -rn "$OUT_ROOT/cpl_0.0/target_s50" "$OUT_ROOT/cpl_$C/target_s50" 2>/dev/null || true
  python analyze_sr.py --out_root "$OUT_ROOT/cpl_$C" --ref_method target_s50
done

echo "=== assembling coupling x tolerance table ==="
python assemble_coupling_table.py \
  --couplings 0.0 0.3 0.6 0.9 \
  --sweep_dirs "$OUT_ROOT/cpl_0.0" "$OUT_ROOT/cpl_0.3" "$OUT_ROOT/cpl_0.6" "$OUT_ROOT/cpl_0.9" \
  --aurc_summaries \
    "${AURC_0:-/dev/null}" "${AURC_3:-/dev/null}" \
    "${AURC_6:-/dev/null}" "${AURC_9:-/dev/null}" \
  --out_dir "$OUT_ROOT/coupling_table"

echo "=== done ==="
echo "  operating-point grid : $OUT_ROOT/coupling_table/coupling_table.{csv,tex}"
echo "  matched-cost targets : $OUT_ROOT/cpl_0.0/sr_table.csv (Target-44/45/48)"
