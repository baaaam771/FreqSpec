#!/bin/bash
# ============================================================================
# Stage 4 — operating-point refinement: split depth m, structured 2x2 token
#           blocks (wall-clock path), re-injection ablation, FID.
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
ROOT=/mnt/HDD_12TB/bam_ki
# DATA_VA=$ROOT/imagenet64/val
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp

# split-depth sweep (m=0 whole-net sparse vs deeper dense prefixes)
for M in 0 3 6; do
  python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dace --selector mask --easy anchor \
    --steps 30 --cache_period 2 --split_m $M --hard_ratio 0.3 \
    --restrict_to_mask --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/split_m${M}
done

# structured 2x2 block sparsity (kernel-friendly hard sets)
python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dace --selector mask --easy anchor \
    --steps 30 --cache_period 2 --split_m 0 --hard_ratio 0.3 --block 2 \
    --restrict_to_mask --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/block2_mask_r0.3

# re-injection ablation (sanity: without it the task degrades to generation)
python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dace --selector mask --easy anchor \
    --steps 30 --cache_period 2 --split_m 0 --hard_ratio 0.3 --no_reinject \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/noreinject_mask_r0.3

python dit_inpaint_assemble.py --root $RES --out $RES/table_stage4

# FID example (clean-fid; on Python 3.14 use workers 0 equivalents):
#   python -c "from cleanfid import fid; \
#     print(fid.compute_fid('$RES/dace_mask_auto_c2/png', '$DATA_VA_FLAT'))"
