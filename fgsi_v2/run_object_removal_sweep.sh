#!/usr/bin/env bash
# run_object_removal_sweep.sh — Semantic object-removal mask evaluation
# on COCO, using instance-segmentation polygons as inpainting masks.
#
# Covers the "object-removal" difficulty stratum requested in review
# feedback (dQQy: complex masks; agent workload stratification). Masks
# are one dilated instance polygon per image (2-40 percent area),
# deterministically chosen per image seed. Targets cannot be reused
# (masks differ from all other runs) — s50/s30 computed fresh.
#
# COST: (2 targets + 2 presets) x 50 imgs x ~30-40s  =~ 4-5 h.
#
# REQUIRES: baseline_sweep.py = patched v4.
set -euo pipefail
grep -q "coco_object" baseline_sweep.py || { echo "ERROR: baseline_sweep.py is not v4"; exit 1; }

# ================= EDIT HERE =================
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"
DRAFT_CKPT="/mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_coco_400k.pt"
DATA_ROOT="/mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017"
CAPTION_JSON="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json"
INSTANCES_JSON="/mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/instances_val2017.json"
OUT="/mnt/HDD_12TB/bam_ki/results/objectremoval_coco"
NUM_IMAGES=50
# =============================================

[[ -f "${DRAFT_CKPT}" ]] || { echo "ERROR: draft ckpt missing: ${DRAFT_CKPT}"; exit 1; }

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
  --fs_presets strict,default \
  --resume \
  --mask_mode coco_object \
  --instances_json "${INSTANCES_JSON}" \
  --save_usage_maps \
  --K 3 --patch 4 --t_spec_start 0.7 --beta 10.0 --boundary_weight 1.0 \
  --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
  --x0_strict_center 0.45 --x0_strict_width 0.12 \
  --blend_temperature 0.10 \
  --mask_interior_weight 0.5 \
  --drift_k_switch_threshold 0.006

echo ""
echo "[objrm] done -> ${OUT}"
echo "[objrm] collect: python collect_sweep_metrics_v2.py --run_dirs ${OUT} \\"
echo "        --methods freqspec_strict freqspec_default --out_prefix objrm"
