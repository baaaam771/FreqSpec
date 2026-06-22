#!/usr/bin/env bash
# run_sr_freqspec.sh — FreqSpec-SR single-image / quick sanity on the server.
#
# AAAI-track super-resolution instantiation of FreqSpec. Reuses the SAME
# generalized verifier (inference/speculative_general.py) as inpainting; only
# the task setup differs (region = whole field, condition = low-res image,
# wavelet-only saliency, no known-region blending).
#
# Target: stabilityai/stable-diffusion-x4-upscaler (frozen, 4ch latent, 7ch UNet)
# Draft : 82.35M U-Net, cond_ch=3 (LR RGB), no mask channel
#
# Usage:
#   ./run_sr_freqspec.sh <hr_image> <draft_ckpt> [out_dir]
#
# Pre-flight (verify on the server with real weights):
#   - unet.config.in_channels == 7
#   - scheduler.config.prediction_type == "epsilon"
#   - vae.config.scaling_factor (~0.08333), VAE decode upsamples x4
#
# Server notes (per project conventions):
#   export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
#   run inside tmux; one command at a time.
set -euo pipefail

HR_IMAGE="$1"
DRAFT_CKPT="$2"
OUT_DIR="${3:-./results_sr}"

python inference/run_sr.py \
  --image "$HR_IMAGE" \
  --draft_ckpt "$DRAFT_CKPT" \
  --target_id stabilityai/stable-diffusion-x4-upscaler \
  --out_dir "$OUT_DIR" \
  --lr_size 128 --scale 4 \
  --num_steps 50 \
  --noise_level 20 \
  --guidance_scale 1.0 \
  --target_dtype fp16 \
  --use_ema_draft \
  --K 3 --patch 4 --t_spec_start 0.7 \
  --tol_low 0.03 --tol_high 0.30 \
  --blend_temperature 0.10 \
  --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
  --x0_strict_center 0.45 --x0_strict_width 0.12 \
  --drift_k_switch 0.006 --k_switch 0.60

echo "[run_sr_freqspec] done -> $OUT_DIR"
