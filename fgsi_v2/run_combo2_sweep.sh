#!/usr/bin/env bash
# run_combo2_sweep.sh — Run baseline_sweep.py with Combo 2 (full system)
# settings, including draft-usage-map saving for qualitative figures.
#
# All five training-free calibration mechanisms are enabled:
#   - Fix 2 (x0 gate)              -> Fix 4 timestep-dependent strictness
#   - Fix 3 (soft blend)           -> blend_temperature = 0.10
#   - Fix 4 (mask-interior)        -> mask_interior_weight = 0.5
#   - Fix 5 (drift-aware K-step)   -> drift_k_switch_threshold = 0.006
#   - Fix 4' (x0 strict-center)    -> x0_strict_center = 0.45
#
# All three tolerance presets (strict / mid / default) are produced
# automatically by baseline_sweep.py's --target_steps + freqspec preset
# loop. So a single call yields:
#   target_s50, target_s30, freqspec_strict, freqspec_mid, freqspec_default
# All with the SAME seed/mask/prompt per image -> paired comparison.
#
# Usage:
#   ./run_combo2_sweep.sh <dataset_name> <data_root> <draft_ckpt> [caption_json]
#
# Examples:
#   ./run_combo2_sweep.sh coco   /mnt/.../coco2017/val2017 \
#       /mnt/.../draft_final.pt \
#       /mnt/.../annotations/captions_val2017.json
#
#   ./run_combo2_sweep.sh places2 /mnt/.../places2 \
#       /mnt/.../draft_step0290000.pt
#
#   ./run_combo2_sweep.sh ffhq    /mnt/.../ffhq \
#       /mnt/.../ffhq_draft_final.pt
#
set -euo pipefail

DATASET="$1"
DATA_ROOT="$2"
DRAFT_CKPT="$3"
CAPTION_JSON="${4:-}"

# === Output dir ===
OUT_ROOT="/mnt/HDD_12TB/bam_ki/results/qualitative_${DATASET}_run100"

# === SDXL target (same for all datasets) ===
TARGET_ID="/mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1"

# === Optional caption flag for COCO ===
CAPTION_ARG=""
PROMPT_ARG="--auto_prompt"   # default: prompt from path (FFHQ/Places2)
if [[ -n "${CAPTION_JSON}" ]]; then
    CAPTION_ARG="--caption_json ${CAPTION_JSON}"
    PROMPT_ARG=""             # COCO captions take priority
fi

# === Sanity print: confirm Combo 2 settings ===
echo ""
echo "================================================================"
echo "[combo2] Running Combo 2 sweep on dataset: ${DATASET}"
echo "================================================================"
echo "  data_root  : ${DATA_ROOT}"
echo "  draft_ckpt : ${DRAFT_CKPT}"
echo "  out_root   : ${OUT_ROOT}"
echo "  target     : ${TARGET_ID}"
[[ -n "${CAPTION_JSON}" ]] && echo "  captions   : ${CAPTION_JSON}" || true
echo ""
echo "  Combo 2 = x0_gate + soft_blend + timestep_strictness +"
echo "            mask_interior + drift_aware_K_step_gating"
echo "  All three tolerance presets (strict, mid, default) generated."
echo "  Draft usage maps are saved for figure assembly."
echo "================================================================"
echo ""

# === Run sweep ===
python baseline_sweep.py \
    --target_id "${TARGET_ID}" \
    --draft_ckpt "${DRAFT_CKPT}" \
    --use_ema_draft \
    --data_root "${DATA_ROOT}" \
    ${CAPTION_ARG} ${PROMPT_ARG} \
    --out_root "${OUT_ROOT}" \
    --num_images 100 \
    --image_size 1024 \
    --target_steps 50 30 \
    \
    `# ---- Combo 2 settings (paper hyperparameter table) ----` \
    --K 3 \
    --patch 4 \
    --t_spec_start 0.7 \
    --beta 10.0 \
    --boundary_weight 1.0 \
    \
    `# Fix 4 (timestep-dependent x0 gate)` \
    --x0_thr_strict 0.02 \
    --x0_thr_loose  0.07 \
    --x0_strict_center 0.45 \
    --x0_strict_width  0.12 \
    \
    `# Fix 3 (soft blend)` \
    --blend_temperature 0.10 \
    \
    `# Fix 4' (mask-interior strictness)` \
    --mask_interior_weight 0.5 \
    \
    `# Fix 5 (drift-aware K-step gating)` \
    --drift_k_switch_threshold 0.006 \
    \
    `# Save usage maps for Figure 5 column 7 (requires patched speculative.py)` \
    --save_usage_maps

echo ""
echo "[combo2] Sweep finished."
echo "[combo2] Outputs in: ${OUT_ROOT}/"
echo "[combo2]   target_s50/   img_NNN/{out,gt,mask}.png"
echo "[combo2]   target_s30/   img_NNN/{out,gt,mask}.png"
echo "[combo2]   freqspec_strict/   img_NNN/{out,gt,mask,usage_map}.png"
echo "[combo2]   freqspec_mid/      img_NNN/{out,gt,mask,usage_map}.png"
echo "[combo2]   freqspec_default/  img_NNN/{out,gt,mask,usage_map}.png"
echo ""
