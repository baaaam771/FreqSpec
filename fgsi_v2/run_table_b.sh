#!/usr/bin/env bash
# ====================================================================
# run_table_b.sh  —  end-to-end Table B (saliency-signal ablation).
#
# Steps (all on ONE shared, seed-fixed COCO manifest so everything is paired):
#   1. target_s50 reference outputs           -> for LPIPS_t
#   2. 9 saliency configs (sweep)             -> out.png + patch logs
#   3. per-config verifier analysis           -> FAR / AURC
#   4. assemble                               -> table_b.csv / table_b.tex
#
# Paths below are filled in for the sdxl_coco_v2 draft. If your SDXL-Inpainting
# checkpoint folder or captions filename differ, edit TARGET_ID / CAPTION_JSON.
# ====================================================================
set -euo pipefail

# ---- paths (edit if needed) ----
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
DRAFT_CKPT="/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_final.pt"
DATA_ROOT="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
CAPTION_JSON="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
OUT_ROOT="/mnt/HDD_12TB/bam_ki/results/saliency_ablation_coco"
NUM_IMAGES=300
IMAGE_SIZE=1024
SEED=42
DEVICE="cuda"
BOUNDARY_K=8
# --------------------------------

CONFIGS=(none_uniform random sobel variance laplacian wavelet_only boundary_only wavelet_boundary full)

echo "==================================================================="
echo "[table_b] STEP 1/4 : target_s50 reference (for LPIPS_t)"
echo "==================================================================="
python run_target_reference.py \
    --target_id "$TARGET_ID" --draft_ckpt "$DRAFT_CKPT" --use_ema_draft \
    --data_root "$DATA_ROOT" --caption_json "$CAPTION_JSON" \
    --out_root "$OUT_ROOT" \
    --num_images "$NUM_IMAGES" --image_size "$IMAGE_SIZE" --seed "$SEED" \
    --device "$DEVICE" --resume

echo "==================================================================="
echo "[table_b] STEP 2/4 : 9 saliency configs (sweep + patch logs)"
echo "==================================================================="
python saliency_ablation_sweep.py \
    --target_id "$TARGET_ID" --draft_ckpt "$DRAFT_CKPT" --use_ema_draft \
    --data_root "$DATA_ROOT" --caption_json "$CAPTION_JSON" \
    --out_root "$OUT_ROOT" \
    --num_images "$NUM_IMAGES" --image_size "$IMAGE_SIZE" --seed "$SEED" \
    --device "$DEVICE" --resume

echo "==================================================================="
echo "[table_b] STEP 3/4 : per-config verifier analysis (FAR / AURC)"
echo "==================================================================="
for cfg in "${CONFIGS[@]}"; do
    logs="$OUT_ROOT/$cfg/patch_logs"
    if [ -d "$logs" ]; then
        echo "  [verif] $cfg"
        python analyze_verifier_reliability.py \
            --logs_dir "$logs" \
            --out_dir  "$OUT_ROOT/verif/$cfg" \
            --bad_quantile 0.8 --random_repeats 20 --bootstrap 1000
    else
        echo "  [verif] SKIP $cfg (no patch_logs)"
    fi
done

echo "==================================================================="
echo "[table_b] STEP 4/4 : assemble Table B"
echo "==================================================================="
python assemble_table_b.py \
    --sweep_root "$OUT_ROOT" \
    --ref_dir    "$OUT_ROOT/target_s50" \
    --verif_root "$OUT_ROOT/verif" \
    --out        "$OUT_ROOT" \
    --far_coverage 0.5 --boundary_k "$BOUNDARY_K" --device "$DEVICE"

echo ""
echo "[table_b] DONE -> $OUT_ROOT/table_b.tex (and table_b.csv)"
echo "[table_b] tip: for a coverage-matched variant, re-run STEP 2 with"
echo "          --tol_low/--tol_high tuned so each config lands near 50% coverage,"
echo "          into a separate OUT_ROOT, then re-assemble."
