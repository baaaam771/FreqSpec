#!/usr/bin/env bash
# run_sr_benchmarks.sh — FreqSpec-SR on standard SR test sets (native resolution).
#
# Set5 / Set14 / BSD100 / Urban100, each evaluated at native HR (cropped to a
# multiple of 8*scale) since these images vary in size and are mostly < 512.
# Produces one operating-point table per dataset, then a combined cross-dataset
# table via assemble_sr_benchmarks.py.
#
# Usage:
#   ./run_sr_benchmarks.sh <bench_root> <draft_ckpt> <out_root>
# where <bench_root> contains Set5_HR/ Set14_HR/ BSD100_HR/ Urban100_HR/
set -euo pipefail

BENCH_ROOT="$1"          # /mnt/HDD_12TB/bam_ki/datasets/sr_bench
DRAFT_CKPT="$2"
OUT_ROOT="$3"
TARGET_ID="stabilityai/stable-diffusion-x4-upscaler"

for DS in Set5 Set14 BSD100 Urban100; do
  echo "=== benchmark: $DS ==="
  python sr_baseline_sweep.py \
    --target_id "$TARGET_ID" --draft_ckpt "$DRAFT_CKPT" --use_ema_draft \
    --data_root "$BENCH_ROOT/${DS}_HR" \
    --out_root  "$OUT_ROOT/$DS" \
    --num_images 100 --scale 4 --native_res \
    --target_steps 50 40 30 --num_steps 50 \
    --blend_temperature 0.10 \
    --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
    --x0_strict_center 0.45 --x0_strict_width 0.12 \
    --drift_k_switch_threshold 0.006 --save_usage_maps

  python analyze_sr.py --out_root "$OUT_ROOT/$DS" --ref_method target_s50
done

echo "=== assembling cross-dataset table ==="
python assemble_sr_benchmarks.py --out_root "$OUT_ROOT" \
  --datasets Set5 Set14 BSD100 Urban100

echo "=== done -> $OUT_ROOT/sr_benchmarks.{csv,tex} ==="
