#!/usr/bin/env bash
# ====================================================================
# run_table_c.sh  —  end-to-end Table C (draft training objective ablation).
#
# UNLIKE Tables A/B, this REQUIRES RE-TRAINING: each of the 9 objective
# variants is a draft trained from scratch. Budget realistically.
#
# Stages:
#   train   : train all 9 variants (run_training_ablation.sh)
#   eval    : per variant -> FreqSpec eval (patch logs + outputs) + verifier
#             analysis (AURC); plus one shared target_s50 reference
#   table   : assemble table_c.csv / table_c.tex
#
# Usage:
#   bash run_table_c.sh train      # long: 9 trainings (use prune first, see below)
#   bash run_table_c.sh eval
#   bash run_table_c.sh table
#   bash run_table_c.sh all        # eval + table (assumes drafts already trained)
#
# Two-stage training is strongly recommended (edit run_training_ablation.sh):
#   bash run_training_ablation.sh prune   # all 9 to 100k steps, pick top 3-4
#   bash run_training_ablation.sh full    # winners to 400k
# ====================================================================
set -euo pipefail

# ---- paths (edit if needed) ----
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
DATA_ROOT="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
CAPTION_JSON="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
TRAIN_BASE="/mnt/HDD_12TB/bam_ki/runs/table_c_ablation"   # where drafts are saved
EVAL_ROOT="/mnt/HDD_12TB/bam_ki/results/table_c_eval"     # where eval lands
NUM_IMAGES=200          # fixed COCO eval set for all variants
IMAGE_SIZE=1024
SEED=42                 # single seed so outputs match target_s50 reference
DEVICE="cuda"
TRAIN_STAGE_SUFFIX="full"   # drafts live in <TRAIN_BASE>/<variant>_<suffix>/draft_final.pt
# --------------------------------

# variant_dir (assemble_table_c.py names)  ->  trained-draft dir name
declare -A VARIANT_TRAINDIR
VARIANT_TRAINDIR["global_gt_only"]="global_gt_only"
VARIANT_TRAINDIR["global_distill_only"]="global_distill_only"
VARIANT_TRAINDIR["hard_gt_only"]="hard_gt_only"
VARIANT_TRAINDIR["easy_distill_hard_gt"]="easy_distill_hard_gt"
VARIANT_TRAINDIR["easy_distill_uniform_gt"]="easy_distill_uniform_gt"
VARIANT_TRAINDIR["hard_gt_uniform_gt"]="hard_gt_uniform_gt"
VARIANT_TRAINDIR["full_region_aware"]="full_region_aware"
VARIANT_TRAINDIR["full_mask_only_Mt"]="full_mask_only_Mt"
VARIANT_TRAINDIR["full_static_M"]="full_static_M"

STAGE="${1:-all}"

if [ "$STAGE" = "train" ]; then
    echo "[table_c] launching all-variant training via run_training_ablation.sh"
    echo "[table_c] (edit that script's paths/budget; prune then full recommended)"
    bash run_training_ablation.sh full
    exit 0
fi

run_eval () {
    # one shared target_s50 reference (for Mask LPIPS_t), using any draft just to
    # share the loader; reference itself only uses the 50-step target.
    local any_draft="${TRAIN_BASE}/full_region_aware_${TRAIN_STAGE_SUFFIX}/draft_final.pt"
    echo "[table_c] target_s50 reference -> ${EVAL_ROOT}/target_s50"
    python run_target_reference.py \
        --target_id "$TARGET_ID" --draft_ckpt "$any_draft" --use_ema_draft \
        --data_root "$DATA_ROOT" --caption_json "$CAPTION_JSON" \
        --out_root "$EVAL_ROOT" \
        --num_images "$NUM_IMAGES" --image_size "$IMAGE_SIZE" --seed "$SEED" \
        --device "$DEVICE" --resume

    for variant in "${!VARIANT_TRAINDIR[@]}"; do
        local draft="${TRAIN_BASE}/${VARIANT_TRAINDIR[$variant]}_${TRAIN_STAGE_SUFFIX}/draft_final.pt"
        if [ ! -f "$draft" ]; then
            echo "[table_c] SKIP $variant (no draft at $draft)"
            continue
        fi
        local vout="${EVAL_ROOT}/${variant}"
        echo "==================================================================="
        echo "[table_c] eval $variant"
        echo "==================================================================="
        python verifier_reliability_sweep.py \
            --target_id "$TARGET_ID" --draft_ckpt "$draft" --use_ema_draft \
            --data_root "$DATA_ROOT" --caption_json "$CAPTION_JSON" \
            --out_root "$vout" \
            --num_images "$NUM_IMAGES" --image_size "$IMAGE_SIZE" \
            --seed "$SEED" --seeds "$SEED" \
            --device "$DEVICE" --save_outputs --resume
        python analyze_verifier_reliability.py \
            --logs_dir "$vout/patch_logs" \
            --out_dir  "$vout/analysis" \
            --bad_quantile 0.8 --random_repeats 20 --bootstrap 1000
    done
}

run_table () {
    python assemble_table_c.py \
        --eval_root "$EVAL_ROOT" \
        --ref_dir   "$EVAL_ROOT/target_s50" \
        --out       "$EVAL_ROOT" \
        --beta 10.0 --boundary_k 8 --device "$DEVICE"
    echo "[table_c] -> ${EVAL_ROOT}/table_c.tex (+ table_c.csv)"
}

case "$STAGE" in
    eval)  run_eval ;;
    table) run_table ;;
    all)   run_eval; run_table ;;
    *) echo "usage: bash run_table_c.sh [train|eval|table|all]"; exit 1 ;;
esac
