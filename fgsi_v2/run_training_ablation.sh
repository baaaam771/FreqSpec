#!/usr/bin/env bash
# ====================================================================
# run_training_ablation.sh  —  Table C (region-aware draft objective).
#
# Re-trains the draft under different loss-term combinations, holding EVERY
# other training factor fixed (same init seed, optimizer steps, images, masks,
# batch size, lr, EMA) so only the objective changes. Two-stage budget per the
# experiment design: train all variants to a short budget to prune, then take
# the top few to the full budget.
#
# Loss terms (all default ON = paper's full region-aware objective):
#   easy-region distillation  : alpha_distill (1-M_t) ||eps_d - eps_target||^2
#   hard-region ground-truth   : gamma_main    M_t     ||eps_d - eps_gt||^2
#   uniform safety             : lambda_uniform        ||eps_d - eps_gt||^2
#
# Toggle flags added to train.py:
#   --no_distill --distill_global --no_hard_gt --no_uniform_gt
#   --mask_signal{wavelet|...} --mask_no_base --static_mask
#
# Edit the paths below, then:   bash run_training_ablation.sh prune
#                          or:  bash run_training_ablation.sh full
# ====================================================================
set -euo pipefail

# ---- EDIT THESE ----
DATA_ROOT="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
CAPTION_JSON="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
OUT_BASE="/mnt/HDD_12TB/bam_ki/runs/table_c_ablation"
BATCH=4
LR=1e-4
# stage budgets (steps)
PRUNE_STEPS=100000
FULL_STEPS=400000
# fixed weights (same as paper full objective)
A=0.5; G=2.0; U=1.0
# --------------------

STAGE="${1:-prune}"
if [ "$STAGE" = "prune" ]; then STEPS=$PRUNE_STEPS; else STEPS=$FULL_STEPS; fi
echo "[table_c] stage=$STAGE steps=$STEPS"

# variant name -> extra train.py flags
declare -A VARIANTS
# VARIANTS["global_gt_only"]="--no_distill --no_hard_gt"                       # uniform GT everywhere only
VARIANTS["global_distill_only"]="--distill_global --no_hard_gt --no_uniform_gt"
# VARIANTS["hard_gt_only"]="--no_distill --no_uniform_gt"
# VARIANTS["easy_distill_hard_gt"]="--no_uniform_gt"
# VARIANTS["easy_distill_uniform_gt"]="--no_hard_gt"
# VARIANTS["hard_gt_uniform_gt"]="--no_distill"
VARIANTS["full_region_aware"]=""                                            # paper default
# extra design-choice variants
# VARIANTS["full_mask_only_Mt"]="--mask_no_base"                              # M_t from mask geometry only
# VARIANTS["full_static_M"]="--static_mask"                                   # timestep-independent M

run_variant () {
  local name="$1"; local flags="$2"
  local out="${OUT_BASE}/${name}_${STAGE}"
  echo ""
  echo "==================================================================="
  echo "[table_c] variant=${name}  flags='${flags}'  -> ${out}"
  echo "==================================================================="
  # already finished? skip.
  if [ -f "${out}/draft_final.pt" ]; then
    echo "[table_c] SKIP ${name}: draft_final.pt already exists."
    return 0
  fi
  # partial run? resume from the latest checkpoint.
  local resume_arg=""
  if [ -f "${out}/draft_latest.pt" ]; then
    echo "[table_c] RESUME ${name} from draft_latest.pt"
    resume_arg="--resume ${out}/draft_latest.pt"
  fi
  python -m training.train \
    --target_id "${TARGET_ID}" \
    --data_root "${DATA_ROOT}" \
    --caption_json "${CAPTION_JSON}" \
    --out_dir "${out}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --max_steps "${STEPS}" \
    --alpha_distill "${A}" --gamma_main "${G}" --lambda_uniform "${U}" \
    ${resume_arg} \
    ${flags}
}

for name in "${!VARIANTS[@]}"; do
  run_variant "${name}" "${VARIANTS[$name]}"
done

echo ""
echo "[table_c] all ${STAGE} runs done under ${OUT_BASE}"
echo "[table_c] next: evaluate each draft with verifier_reliability_sweep.py +"
echo "          analyze_verifier_reliability.py (draft eps-MSE / x0-MSE,"
echo "          coverage, target NFE, Mask LPIPS_t, AURC) on a FIXED COCO set."