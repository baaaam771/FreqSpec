#!/usr/bin/env bash
# run_large_mask_sweep.sh — Large/complex-mask robustness evaluation.
#
# Reviewer request (dQQy): "the experiments mainly consider small masks.
# The method should also be evaluated using large and complex masks."
#
# Runs the Combo 2 system with --mask_mode large (40-60% coverage masks)
# on all 3 datasets. Note: target baselines CANNOT be reused here (masks
# differ from run100), so target_s50 / target_s30 are recomputed.
#
# COST: 3 datasets x (2 targets + 2 presets) x 50 imgs x ~40s = ~13 h.
#
# Usage:  ./run_large_mask_sweep.sh            # all 3 datasets
#         ./run_large_mask_sweep.sh coco       # one dataset
set -euo pipefail
grep -q "default_prompt" baseline_sweep.py || { echo "ERROR: baseline_sweep.py is not v3"; exit 1; }

# ================= EDIT HERE =================
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
RESULTS_ROOT="/mnt/HDD_12TB/bam_ki/results"

declare -A DATA_ROOT=(
  [ffhq]="/mnt/HDD_12TB/bam_ki/datasets/ffhq_hf/images"
  [places2]="/mnt/HDD_12TB/bam_ki/datasets/places2"
  [coco]="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
)
declare -A CAPTION_JSON=(
  [ffhq]=""
  [places2]=""
  [coco]="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
)
declare -A DRAFT_CKPT=(
  [ffhq]="/mnt/HDD_12TB/bam_ki/runs/sdxl_ffhq_v1/draft_ffhq_400k.pt"
  [places2]="/mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_places2_400k.pt"
  [coco]="/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_coco_400k.pt"
)
NUM_IMAGES=50
COV_LO=0.40
COV_HI=0.60
# =============================================

run_ds () {
  local DS="$1"
  local OUT="${RESULTS_ROOT}/largemask_${DS}"
  echo ""
  echo "=============================================================="
  echo "[largemask] dataset: ${DS}  coverage: ${COV_LO}-${COV_HI}"
  echo "[largemask] out: ${OUT}"
  echo "=============================================================="
  local CAP_ARG=""
  local PROMPT_ARG="--auto_prompt"
  if [[ -n "${CAPTION_JSON[${DS}]}" ]]; then
    CAP_ARG="--caption_json ${CAPTION_JSON[${DS}]}"
    PROMPT_ARG=""
  elif [[ "${DS}" == "ffhq" ]]; then
    PROMPT_ARG="--default_prompt \"a photo of a person\""
  fi
  eval python baseline_sweep.py \
    --target_id "${TARGET_ID}" \
    --draft_ckpt "${DRAFT_CKPT[${DS}]}" \
    --use_ema_draft \
    --data_root "${DATA_ROOT[${DS}]}" \
    ${CAP_ARG} ${PROMPT_ARG} \
    --out_root "${OUT}" \
    --num_images "${NUM_IMAGES}" \
    --image_size 1024 \
    --target_steps 50 30 \
    --fs_presets strict,default \
    --resume \
    --mask_mode large \
    --large_coverage ${COV_LO} ${COV_HI} \
    --save_usage_maps \
    --K 3 --patch 4 --t_spec_start 0.7 --beta 10.0 --boundary_weight 1.0 \
    --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
    --x0_strict_center 0.45 --x0_strict_width 0.12 \
    --blend_temperature 0.10 \
    --mask_interior_weight 0.5 \
    --drift_k_switch_threshold 0.006
}

if [[ $# -eq 1 ]]; then
  run_ds "$1"
else
  for DS in coco places2 ffhq; do
    run_ds "${DS}"
  done
fi

echo ""
echo "[largemask] done. Collect with:"
echo "  python collect_sweep_metrics.py --run_dirs ${RESULTS_ROOT}/largemask_* \\"
echo "      --methods freqspec_strict freqspec_default --out_csv largemask.csv"
echo ""
echo "[largemask] qualitative figure (reuses the 8-col assembler):"
echo "  python assemble_qualitative_3datasets.py --ffhq_dir ${RESULTS_ROOT}/largemask_ffhq \\"
echo "      --places2_dir ${RESULTS_ROOT}/largemask_places2 --coco_dir ${RESULTS_ROOT}/largemask_coco \\"
echo "      --ffhq_idx N --places2_idx N --coco_idx N --out_path ${RESULTS_ROOT}/fig_largemask"