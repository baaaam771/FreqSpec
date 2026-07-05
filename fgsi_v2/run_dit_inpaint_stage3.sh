#!/bin/bash
# ============================================================================
# Stage 3 — selector ablation + easy-token source ablation.
#   selectors: mask / boundary / freq / delta / combo / anchor / random /
#              oracle (diagnostic bound)
#   easy: anchor (target-eps reuse) vs draft (verifier regime)
#   The DACE finding to test: does selection stay INERT (as in generation)
#   or does the mask/boundary/delta structure make it matter again?
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
ROOT=/mnt/HDD_12TB/bam_ki
# DATA_VA=$ROOT/imagenet64/val
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp

# selector ablation, target-eps reuse, r=0.3, c=2, S=30
for SEL in mask boundary freq delta combo random oracle; do
  python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dace --selector $SEL --easy anchor \
    --steps 30 --cache_period 2 --split_m 0 --hard_ratio 0.3 \
    --restrict_to_mask \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/sel_${SEL}_r0.3_reuse
done

# anchor selector + draft-easy (verifier regime with a real draft)
for SEL in anchor mask random; do
  python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --draft $CKPT/draft_nano.pt --draft_model DiT-Nano-Inp \
    --data_root $DATA_VA --mode dace --selector $SEL --easy draft \
    --steps 30 --cache_period 2 --split_m 0 --hard_ratio 0.3 \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/sel_${SEL}_r0.3_drafteasy
done

# mixing ceiling reference (both dense)
python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --draft $CKPT/draft_nano.pt --draft_model DiT-Nano-Inp \
    --data_root $DATA_VA --mode mix --steps 30 --hard_ratio 0.3 \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/mix_ceiling_r0.3

python dit_inpaint_assemble.py --root $RES --out $RES/table_stage3
