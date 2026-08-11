#!/usr/bin/env bash
# run_phase2_remeasure.sh — Main-table re-measurement with 400k drafts,
# dense target-step brackets (for exact speed matching), and the
# corrected FFHQ evaluation prompt.
#
# WHY DENSE STEPS: reviewers' core objection is the matched-speed
# comparison vs reduced-step SDXL. With only {50,30}, speed matching is
# coarse. Dense brackets {50,48,46,44,42,40,36,30} let us interpolate the
# exact reduced-step operating point matching each FreqSpec preset, and
# double as the per-image latency/quality log for budget-policy
# simulation (fixed-step vs static vs budget-aware — no extra GPU).
#
# PROMPT FIX (critical): FFHQ drafts were TRAINED with default_prompt
# "a photo of a person", but earlier sweeps evaluated with --auto_prompt
# (path-derived, wrong for ffhq_hf/images) falling back to "a photograph".
# => FFHQ is re-run FROM SCRATCH here (targets included — the prompt
#    changes the target trajectory, so old targets are not reusable).
# Places2 (--auto_prompt, matches training) and COCO (captions, matches
# training) keep their prompt pipelines; their s50/s30 targets are
# symlinked from run100 and skipped via --resume.
#
# REQUIRES: baseline_sweep.py = patched v3 (checked below).
#
# COST (rough, ~28s/img at 1024 for 50 steps, scaling with steps):
#   FFHQ   : 8 targets + 3 presets, all fresh  ≈ 7-8 h
#   Places2: 6 new targets + 3 presets          ≈ 6-7 h
#   COCO   : 6 new targets + 3 presets          ≈ 6-7 h
#   Total  ≈ 20 h  (one long day / overnight x2)
#
# Usage: ./run_phase2_remeasure.sh            # all three
#        ./run_phase2_remeasure.sh ffhq       # one dataset
set -euo pipefail

# ---------- guard: patched v3 installed? ----------
grep -q "default_prompt" baseline_sweep.py || {
  echo "ERROR: baseline_sweep.py lacks --default_prompt (v3 not installed)"; exit 1; }
grep -q "fs_presets" baseline_sweep.py || {
  echo "ERROR: baseline_sweep.py lacks --fs_presets (v2/v3 not installed)"; exit 1; }

# ================= EDIT HERE =================
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
RESULTS_ROOT="/mnt/HDD_12TB/bam_ki/results"
NUM_IMAGES=100
TARGET_STEPS_DENSE="50 48 46 44 42 40 36 30"

declare -A DATA_ROOT=(
  [ffhq]="/mnt/HDD_12TB/bam_ki/datasets/ffhq_hf/images"
  [places2]="/mnt/HDD_12TB/bam_ki/datasets/places2"
  [coco]="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
)
declare -A DRAFT_CKPT=(     # 400k unified drafts (safety copies)
  [ffhq]="/mnt/HDD_12TB/bam_ki/runs/sdxl_ffhq_v1/draft_ffhq_400k.pt"
  [places2]="/mnt/HDD_12TB/bam_ki/runs/sdxl_v1/draft_places2_400k.pt"
  [coco]="/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_coco_400k.pt"
)
declare -A OLD_RUN=(        # run100 dirs (target s50/s30 reuse source)
  [ffhq]=""                 # EMPTY on purpose: FFHQ targets invalid (prompt bug)
  [places2]="${RESULTS_ROOT}/qualitative_places2_run100"
  [coco]="${RESULTS_ROOT}/qualitative_coco_run100"
)
# =============================================

prompt_args () {
  case "$1" in
    ffhq)    echo "--default_prompt \"a photo of a person\"" ;;
    places2) echo "--auto_prompt" ;;
    coco)    echo "--caption_json /mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json" ;;
  esac
}

run_ds () {
  local DS="$1"
  local OUT="${RESULTS_ROOT}/main_${DS}_400k"
  local CKPT="${DRAFT_CKPT[${DS}]}"
  [[ -f "${CKPT}" ]] || { echo "ERROR: draft ckpt missing: ${CKPT} — make the 400k safety copy first"; exit 1; }

  echo ""
  echo "=============================================================="
  echo "[phase2] ${DS}  draft=${CKPT}"
  echo "[phase2] out: ${OUT}"
  echo "=============================================================="
  mkdir -p "${OUT}"

  # Reuse s50/s30 targets where the prompt pipeline is unchanged
  if [[ -n "${OLD_RUN[${DS}]}" ]]; then
    for TM in target_s50 target_s30; do
      if [[ ! -e "${OUT}/${TM}" && -d "${OLD_RUN[${DS}]}/${TM}" ]]; then
        ln -s "${OLD_RUN[${DS}]}/${TM}" "${OUT}/${TM}"
        echo "[phase2] symlinked ${TM} (skipped via --resume)"
      fi
    done
  else
    echo "[phase2] ${DS}: NO target reuse (prompt fix) — all targets fresh"
  fi

  eval python baseline_sweep.py \
    --target_id "${TARGET_ID}" \
    --draft_ckpt "${CKPT}" \
    --use_ema_draft \
    --data_root "${DATA_ROOT[${DS}]}" \
    $(prompt_args "${DS}") \
    --out_root "${OUT}" \
    --num_images "${NUM_IMAGES}" \
    --image_size 1024 \
    --target_steps ${TARGET_STEPS_DENSE} \
    --fs_presets strict,mid,default \
    --resume \
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
  for DS in ffhq places2 coco; do run_ds "${DS}"; done
fi

echo ""
echo "[phase2] done. Old run100 dirs (200k/290k drafts) are preserved —"
echo "[phase2] use them with collect_sweep_metrics for the training-budget table."
