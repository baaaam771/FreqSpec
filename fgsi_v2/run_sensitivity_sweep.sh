#!/usr/bin/env bash
# run_sensitivity_sweep.sh — 1-D sensitivity analysis around the Combo 2
# operating point, on ONE dataset (default: COCO = hardest case).
#
# Reviewer requests:
#   e5wk: "relies on many manually selected thresholds ... limited
#          sensitivity analysis"; "sensitivity analysis for the most
#          important thresholds or patch size would strengthen the rebuttal"
#   Zpex: "sensitivity discussion for key hyper-parameters (K, drift
#          threshold, saliency weights)"
#
# Design: hold Combo 2 fixed, vary ONE axis at a time:
#   K                    : 2, [3], 4, 6
#   patch                : 2, [4], 8
#   blend_temperature    : 0.05, [0.10], 0.20
#   drift_k_switch_thr   : 0.004, [0.006], 0.008, off
#   mask_interior_weight : 0.0, [0.5], 0.8
#   x0 gate (strict,loose): (0.01,0.05), [(0.02,0.07)], (0.03,0.10)
#   [..] = Combo 2 center, run once as "center"
#
# 15 runs x 1 preset (default) x NUM_IMAGES(50) x ~45s  = ~9.5 h
# Targets are symlinked from the in-domain run (skipped via --resume).
#
# Usage:  ./run_sensitivity_sweep.sh            # all axes
#         ./run_sensitivity_sweep.sh K          # one axis (K|patch|blend|drift|interior|x0gate|center)
set -euo pipefail

# ================= EDIT HERE =================
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
DRAFT_CKPT="/mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_final.pt"     # COCO draft
DATA_ROOT="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
CAPTION_JSON="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
INDOMAIN_RUN="/mnt/HDD_12TB/bam_ki/results/qualitative_coco_run100"
OUT_BASE="/mnt/HDD_12TB/bam_ki/results/sens_coco"
NUM_IMAGES=50
# =============================================

# Combo 2 center values
C_K=3; C_PATCH=4; C_BLEND=0.10; C_DRIFT=0.006; C_INTERIOR=0.5
C_X0S=0.02; C_X0L=0.07

run_one () {
  local NAME="$1"; shift
  local OUT="${OUT_BASE}_${NAME}"
  echo ""
  echo "=========================================="
  echo "[sens] ${NAME}  ->  ${OUT}"
  echo "=========================================="
  mkdir -p "${OUT}"
  for TM in target_s50 target_s30; do
    if [[ ! -e "${OUT}/${TM}" && -d "${INDOMAIN_RUN}/${TM}" ]]; then
      ln -s "${INDOMAIN_RUN}/${TM}" "${OUT}/${TM}"
    fi
  done
  python baseline_sweep.py \
    --target_id "${TARGET_ID}" \
    --draft_ckpt "${DRAFT_CKPT}" \
    --use_ema_draft \
    --data_root "${DATA_ROOT}" \
    --caption_json "${CAPTION_JSON}" \
    --out_root "${OUT}" \
    --num_images "${NUM_IMAGES}" \
    --image_size 1024 \
    --target_steps 50 30 \
    --fs_presets default \
    --resume \
    --t_spec_start 0.7 --beta 10.0 --boundary_weight 1.0 \
    --x0_strict_center 0.45 --x0_strict_width 0.12 \
    "$@"
}

# Center config args (each axis run overrides ONE of these)
center_args () {
  echo "--K ${C_K} --patch ${C_PATCH} --blend_temperature ${C_BLEND} \
        --drift_k_switch_threshold ${C_DRIFT} \
        --mask_interior_weight ${C_INTERIOR} \
        --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}"
}

AXIS="${1:-all}"

if [[ "$AXIS" == "center" || "$AXIS" == "all" ]]; then
  run_one "center" $(center_args)
fi

if [[ "$AXIS" == "K" || "$AXIS" == "all" ]]; then
  for V in 2 4 6; do
    run_one "K${V}" --K ${V} --patch ${C_PATCH} \
      --blend_temperature ${C_BLEND} --drift_k_switch_threshold ${C_DRIFT} \
      --mask_interior_weight ${C_INTERIOR} \
      --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}
  done
fi

if [[ "$AXIS" == "patch" || "$AXIS" == "all" ]]; then
  for V in 2 8; do
    run_one "patch${V}" --K ${C_K} --patch ${V} \
      --blend_temperature ${C_BLEND} --drift_k_switch_threshold ${C_DRIFT} \
      --mask_interior_weight ${C_INTERIOR} \
      --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}
  done
fi

if [[ "$AXIS" == "blend" || "$AXIS" == "all" ]]; then
  for V in 0.05 0.20; do
    run_one "blend${V}" --K ${C_K} --patch ${C_PATCH} \
      --blend_temperature ${V} --drift_k_switch_threshold ${C_DRIFT} \
      --mask_interior_weight ${C_INTERIOR} \
      --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}
  done
fi

if [[ "$AXIS" == "drift" || "$AXIS" == "all" ]]; then
  for V in 0.004 0.008; do
    run_one "drift${V}" --K ${C_K} --patch ${C_PATCH} \
      --blend_temperature ${C_BLEND} --drift_k_switch_threshold ${V} \
      --mask_interior_weight ${C_INTERIOR} \
      --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}
  done
  # drift gate OFF (flag omitted -> None -> disabled)
  run_one "driftoff" --K ${C_K} --patch ${C_PATCH} \
    --blend_temperature ${C_BLEND} \
    --mask_interior_weight ${C_INTERIOR} \
    --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}
fi

if [[ "$AXIS" == "interior" || "$AXIS" == "all" ]]; then
  for V in 0.0 0.8; do
    run_one "interior${V}" --K ${C_K} --patch ${C_PATCH} \
      --blend_temperature ${C_BLEND} --drift_k_switch_threshold ${C_DRIFT} \
      --mask_interior_weight ${V} \
      --x0_thr_strict ${C_X0S} --x0_thr_loose ${C_X0L}
  done
fi

if [[ "$AXIS" == "x0gate" || "$AXIS" == "all" ]]; then
  run_one "x0tight" --K ${C_K} --patch ${C_PATCH} \
    --blend_temperature ${C_BLEND} --drift_k_switch_threshold ${C_DRIFT} \
    --mask_interior_weight ${C_INTERIOR} \
    --x0_thr_strict 0.01 --x0_thr_loose 0.05
  run_one "x0loose" --K ${C_K} --patch ${C_PATCH} \
    --blend_temperature ${C_BLEND} --drift_k_switch_threshold ${C_DRIFT} \
    --mask_interior_weight ${C_INTERIOR} \
    --x0_thr_strict 0.03 --x0_thr_loose 0.10
fi

echo ""
echo "[sens] done. Collect with:"
echo "  python collect_sweep_metrics.py --run_dirs ${OUT_BASE}_* \\"
echo "      --methods freqspec_default --out_csv sensitivity_coco.csv"
