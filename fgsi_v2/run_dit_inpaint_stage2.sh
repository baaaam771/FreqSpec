#!/bin/bash
# ============================================================================
# Stage 2 — the deciding quantity + mask-only sparse execution.
#   (a) heterogeneity test: does inpainting supply factor (i)
#       (step-reduction sensitivity in the hole) on top of factor (ii)
#       (concentration aligned with the mask)?  <- DACE forward prediction
#   (b) mask-only DACE (hard = dilated hole tokens, target-eps reuse,
#       draft-free) vs random at the same budget vs dense reduced-step.
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
ROOT=/mnt/HDD_12TB/bam_ki
# DATA_VA=$ROOT/imagenet64/val
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp

# (a) deciding quantity — run FIRST; if the mask-restricted step-reduction
#     curve is flat, the sweep below will reproduce the IN-64 null result.
python dit_inpaint_heterogeneity.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --n_traj 64 --batch 32 \
    --out $RES/heterogeneity

# (b) mask-only routing, target-eps reuse (fully draft-free)
for C in 2 3 5; do
  for SEL in mask random; do
    python dit_inpaint_sampler.py \
      --target $CKPT/target.pt --target_model DiT-S-Inp \
      --data_root $DATA_VA --mode dace --selector $SEL --easy anchor \
      --steps 30 --cache_period $C --split_m 0 --hard_ratio 0 \
      --n_samples 200 --batch 32 --ref_dir $RES/ref \
      --out_dir $RES/dace_${SEL}_auto_c${C}
  done
done

# fixed-budget comparison at r = 0.2 / 0.3 / 0.5
for R in 0.2 0.3 0.5; do
  for SEL in mask random; do
    python dit_inpaint_sampler.py \
      --target $CKPT/target.pt --target_model DiT-S-Inp \
      --data_root $DATA_VA --mode dace --selector $SEL --easy anchor \
      --steps 30 --cache_period 2 --split_m 0 --hard_ratio $R \
      --n_samples 200 --batch 32 --ref_dir $RES/ref \
      --out_dir $RES/dace_${SEL}_r${R}_c2
  done
done

python dit_inpaint_assemble.py --root $RES --out $RES/table_stage2
