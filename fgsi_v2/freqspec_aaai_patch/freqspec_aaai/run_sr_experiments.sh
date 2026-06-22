#!/usr/bin/env bash
# run_sr_experiments.sh — full AAAI SR experiment pipeline on the server.
#
#   1. operating-point sweep  (target_sN vs FreqSpec presets)   -> sr_table.{csv,tex}
#   2. verifier-reliability    (patch logs)                      -> AURC / risk-coverage
#
# Both reuse the task-agnostic verifier (freqspec_core) and, for (2), the SAME
# analyzer as inpainting (analyze_verifier_reliability.py) with no changes.
#
# Usage:
#   ./run_sr_experiments.sh <data_root> <draft_ckpt> <out_root> [num_images]
#
# Server notes:
#   export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp ; run inside tmux.
set -euo pipefail

DATA_ROOT="$1"
DRAFT_CKPT="$2"
OUT_ROOT="$3"
NUM_IMAGES="${4:-100}"
TARGET_ID="stabilityai/stable-diffusion-x4-upscaler"

echo "=== [1/2] SR operating-point sweep ==="
python sr_baseline_sweep.py \
  --target_id "$TARGET_ID" --draft_ckpt "$DRAFT_CKPT" --use_ema_draft \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT/sweep" \
  --num_images "$NUM_IMAGES" --lr_size 128 --scale 4 \
  --target_steps 50 40 30 --num_steps 50 \
  --blend_temperature 0.10 \
  --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
  --x0_strict_center 0.45 --x0_strict_width 0.12 \
  --drift_k_switch_threshold 0.006 --save_usage_maps

python analyze_sr.py --out_root "$OUT_ROOT/sweep" --ref_method target_s50

echo "=== [2/2] SR verifier-reliability sweep ==="
python sr_verifier_reliability_sweep.py \
  --target_id "$TARGET_ID" --draft_ckpt "$DRAFT_CKPT" --use_ema_draft \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT/reliability" \
  --num_images "$NUM_IMAGES" --seeds 0 1 2 3 4 5 --lr_size 128 --scale 4

python analyze_verifier_reliability.py \
  --logs_dir "$OUT_ROOT/reliability/patch_logs" \
  --out_dir  "$OUT_ROOT/reliability/analysis"

echo "=== done -> $OUT_ROOT ==="
echo "  operating-point table : $OUT_ROOT/sweep/sr_table.{csv,tex}"
echo "  reliability (Table 5) : $OUT_ROOT/reliability/analysis/table_a.tex"
