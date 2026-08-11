#!/usr/bin/env bash
# run_cross_dataset_sweep.sh — Cross-dataset draft generalization (3x3 matrix)
#
# Reviewer request (e5wk + Zpex): "A separate draft is trained for each
# dataset ... cross-dataset generalization performance is not reported."
#
# This script evaluates each draft on each OTHER dataset (6 off-diagonal
# cells). The 3 diagonal (in-domain) cells already exist as
# qualitative_{dataset}_run100. All runs share the SAME manifest per
# dataset (same seed/masks/prompts), so cross-draft comparisons are paired.
#
# EFFICIENCY: target_s50 / target_s30 results depend only on the dataset,
# not the draft. We symlink them from the existing in-domain run and use
# --resume so baseline_sweep skips them entirely (reads results.csv).
#
# COST ESTIMATE: 6 runs x 2 presets (strict,default) x 100 imgs x ~40s
#                = ~13 h total. Reduce NUM_IMAGES to 50 to halve.
#
# Usage:
#     ./run_cross_dataset_sweep.sh            # run all 6 off-diagonal
#     ./run_cross_dataset_sweep.sh ffhq coco  # run one cell: coco-draft on ffhq data
set -euo pipefail
grep -q "default_prompt" baseline_sweep.py || { echo "ERROR: baseline_sweep.py is not v3"; exit 1; }

# ================= EDIT HERE: paths =================
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
RESULTS_ROOT="/mnt/HDD_12TB/bam_ki/results"

# Dataset image roots (must MATCH the in-domain run100 sweeps exactly,
# so the seeded manifest picks identical images/masks)
declare -A DATA_ROOT=(
  [ffhq]="/mnt/HDD_12TB/bam_ki/datasets/ffhq_hf/images"
  [places2]="/mnt/HDD_12TB/bam_ki/datasets/places2"
  [coco]="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
)
# COCO captions (empty for others)
declare -A CAPTION_JSON=(
  [ffhq]=""
  [places2]=""
  [coco]="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
)
# Draft checkpoints per TRAINING domain  << EDIT to actual paths
declare -A DRAFT_CKPT=(
  [ffhq]="/mnt/HDD_12TB/bam_ki/runs/sdxl_ffhq_v1/draft_ffhq_400k.pt"
  [places2]="/mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_places2_400k.pt"
  [coco]="/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_coco_400k.pt"
)
# Existing in-domain sweeps (for target-dir reuse)
# Target reuse source = Phase-2 dirs (run AFTER run_phase2_remeasure.sh:
# their targets use the CORRECTED prompt pipeline, esp. FFHQ)
declare -A INDOMAIN_RUN=(
  [ffhq]="${RESULTS_ROOT}/main_ffhq_400k"
  [places2]="${RESULTS_ROOT}/main_places2_400k"
  [coco]="${RESULTS_ROOT}/main_coco_400k"
)
NUM_IMAGES=100          # must match run100 for manifest/idx alignment
FS_PRESETS="strict,default"
# ====================================================

run_cell () {
  local EVAL_DS="$1"    # dataset the images come from
  local DRAFT_DS="$2"   # domain the draft was trained on
  if [[ "${EVAL_DS}" == "${DRAFT_DS}" ]]; then
    echo "[cross] skip diagonal ${EVAL_DS}x${DRAFT_DS} (use in-domain run100)"
    return 0
  fi
  local OUT="${RESULTS_ROOT}/cross_${EVAL_DS}_draft-${DRAFT_DS}"
  echo ""
  echo "=============================================================="
  echo "[cross] eval data: ${EVAL_DS}   draft trained on: ${DRAFT_DS}"
  echo "[cross] out: ${OUT}"
  echo "=============================================================="
  mkdir -p "${OUT}"

  # Reuse target baselines from the in-domain run via symlink + --resume
  for TM in target_s50 target_s30; do
    if [[ ! -e "${OUT}/${TM}" ]]; then
      if [[ -d "${INDOMAIN_RUN[${EVAL_DS}]}/${TM}" ]]; then
        ln -s "${INDOMAIN_RUN[${EVAL_DS}]}/${TM}" "${OUT}/${TM}"
        echo "[cross] symlinked ${TM} from in-domain run (skipped via --resume)"
      else
        echo "[cross] WARNING: ${INDOMAIN_RUN[${EVAL_DS}]}/${TM} missing — target will be recomputed"
      fi
    fi
  done

  local CAP_ARG=""
  local PROMPT_ARG="--auto_prompt"
  if [[ -n "${CAPTION_JSON[${EVAL_DS}]}" ]]; then
    CAP_ARG="--caption_json ${CAPTION_JSON[${EVAL_DS}]}"
    PROMPT_ARG=""
  elif [[ "${EVAL_DS}" == "ffhq" ]]; then
    # FFHQ: training used default_prompt, not path-derived prompts
    PROMPT_ARG="--default_prompt \"a photo of a person\""
  fi

  eval python baseline_sweep.py \
    --target_id "${TARGET_ID}" \
    --draft_ckpt "${DRAFT_CKPT[${DRAFT_DS}]}" \
    --use_ema_draft \
    --data_root "${DATA_ROOT[${EVAL_DS}]}" \
    ${CAP_ARG} ${PROMPT_ARG} \
    --out_root "${OUT}" \
    --num_images "${NUM_IMAGES}" \
    --image_size 1024 \
    --target_steps 50 30 \
    --fs_presets "${FS_PRESETS}" \
    --resume \
    --K 3 --patch 4 --t_spec_start 0.7 --beta 10.0 --boundary_weight 1.0 \
    --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
    --x0_strict_center 0.45 --x0_strict_width 0.12 \
    --blend_temperature 0.10 \
    --mask_interior_weight 0.5 \
    --drift_k_switch_threshold 0.006
}

if [[ $# -eq 2 ]]; then
  run_cell "$1" "$2"
else
  # All 6 off-diagonal cells, grouped by eval dataset (model loads amortize
  # poorly across runs; each run reloads SDXL ~2 min, acceptable)
  for EVAL in ffhq places2 coco; do
    for DRAFT in ffhq places2 coco; do
      run_cell "${EVAL}" "${DRAFT}"
    done
  done
fi

echo ""
echo "[cross] all requested cells done."
echo "[cross] next: python collect_sweep_metrics.py --help"