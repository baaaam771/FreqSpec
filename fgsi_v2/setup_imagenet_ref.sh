#!/usr/bin/env bash
# ============================================================================
# setup_imagenet_ref.sh — download ImageNet-1k val and build 256/512 FID refs.
#
# Produces DiT-eval-matched reference folders for clean-fid:
#   $DEST/imagenet256_val/  (50k PNGs, 256x256)
#   $DEST/imagenet512_val/  (50k PNGs, 512x512)
# Preprocessing matches the official DiT/ADM pipeline: center-crop to the
# short side, then resize to target with bicubic (PIL), which is what the
# published FID numbers assume.
#
# Requires a HuggingFace account + accepted ImageNet-1k license:
#   https://huggingface.co/datasets/ILSVRC/imagenet-1k  (click "Agree")
#   hf auth login    # paste a token with read access
# ============================================================================
set -e
DEST=${DEST:-/mnt/HDD_12TB/bam_ki/datasets}
RAW=$DEST/imagenet_val_raw
export HF_HOME=${HF_HOME:-$DEST/hf_cache}

echo "=== 1. download ImageNet-1k validation split (~6.3 GB) ==="
# validation parquet shards only (real names: validation-00000-of-00014.parquet)
hf download ILSVRC/imagenet-1k --repo-type dataset \
  --include "data/validation-*.parquet" \
  --local-dir $RAW

echo "=== 2. extract + preprocess to 256 and 512 ==="
python preprocess_imagenet_ref.py \
  --raw_dir $RAW \
  --out256 $DEST/imagenet256_val \
  --out512 $DEST/imagenet512_val \
  --num 50000

echo "=== done ==="
echo "REF256=$DEST/imagenet256_val"
echo "REF512=$DEST/imagenet512_val"
