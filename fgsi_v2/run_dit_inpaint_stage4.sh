#!/bin/bash
# ============================================================================
# Stage 4 (REVISED) — cache/reinject 4-way ablation (item 14), block-STRUCTURED
# selection (item 6), split depth, latency (item 10), 3-seed final (item 11).
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
ROOT=/mnt/HDD_12TB/bam_ki
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp
REF=$RES/ref
COMMON="--target $CKPT/target.pt --target_model DiT-S-Inp --data_root $DATA_VA \
        --region mask --budget ratio --selector mask --easy anchor --steps 30 \
        --cache_period 2 --split_m 0 --hard_ratio 0.3 --n_samples 200 \
        --batch 32 --ref_dir $REF"

# ---- (item 14) cache x reinject 4-way ----
python -u dit_inpaint_sampler.py $COMMON --mode dace --suffix cache \
  --out_dir $RES/abl_cache_reinj
python -u dit_inpaint_sampler.py $COMMON --mode dace --suffix cache --no_reinject \
  --out_dir $RES/abl_cache_noreinj
python -u dit_inpaint_sampler.py $COMMON --mode dace --suffix frozen \
  --out_dir $RES/abl_frozen_reinj
python -u dit_inpaint_sampler.py $COMMON --mode dace --suffix frozen --no_reinject \
  --out_dir $RES/abl_frozen_noreinj

# ---- (item 6) token-wise vs 2x2 vs 4x4 block-structured selection ----
for BLK in 1 2 4; do
  python -u dit_inpaint_sampler.py $COMMON --mode dace --suffix cache --block $BLK \
    --out_dir $RES/blockstruct_${BLK}
done

# ---- split depth m ----
for M in 0 3 6; do
  python -u dit_inpaint_sampler.py $COMMON --mode dace --suffix cache --split_m $M \
    --out_dir $RES/split_m${M}
done

# ---- (item 10) latency profiling ----
python -u dit_inpaint_latency.py \
  --target $CKPT/target.pt --target_model DiT-S-Inp \
  --batches 1 4 8 16 --hard_ratio 0.3 --cache_period 2 --split_m 0 \
  --out $RES/latency

# ---- (item 11) 3-seed final on the headline config (larger n) ----
for S in 0 1 2; do
  python -u dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp --data_root $DATA_VA \
    --mode dace --suffix cache --region mask --budget mask_exact \
    --selector mask --easy anchor --steps 30 --cache_period 2 --split_m 0 \
    --n_samples 1000 --batch 32 --run_seed $S --ref_dir $RES/ref_seed$S \
    --out_dir $RES/final_maskexact_seed$S
done

python -u dit_inpaint_assemble.py --root $RES --out $RES/table_stage4 --by_bucket
