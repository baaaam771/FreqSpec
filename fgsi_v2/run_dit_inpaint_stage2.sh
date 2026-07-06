#!/bin/bash
# ============================================================================
# Stage 2 (REVISED) — deciding quantity + mask-only sparse execution.
# Stage 1 (dense/reduced-step baseline)은 재실행하지 않음. --ref_dir 공유로
# dense-50 reference 캐시 재사용. Frequency/selector 수정은 여기서부터 반영.
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
ROOT=/mnt/HDD_12TB/bam_ki
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp
REF=$RES/ref     # Stage 1이 채워둔 dense-50 reference (동일 seed/mask/data)

# (a) deciding quantity — 먼저 실행. in_out_ratio>>1 & mask-restricted
#     step-reduction curve가 가파르면 논문의 forward prediction 성립.
python dit_inpaint_heterogeneity.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --n_traj 64 --batch 32 \
    --out $RES/heterogeneity

# (b) EXACT mask-only (item 3), target-eps reuse (draft-free), c 스윕
for C in 2 3 5; do
  python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dace --suffix cache \
    --region mask --budget mask_exact --selector mask --easy anchor \
    --steps 30 --cache_period $C --split_m 0 \
    --n_samples 200 --batch 32 --ref_dir $REF \
    --out_dir $RES/maskexact_c${C}
done

# (c) fixed-budget mask-restricted mask vs random (item 2, region=mask)
for R in 0.2 0.3 0.5; do
  for SEL in mask random; do
    python dit_inpaint_sampler.py \
      --target $CKPT/target.pt --target_model DiT-S-Inp \
      --data_root $DATA_VA --mode dace --suffix cache \
      --region mask --budget ratio --selector $SEL --easy anchor \
      --steps 30 --cache_period 2 --split_m 0 --hard_ratio $R \
      --n_samples 200 --batch 32 --ref_dir $REF \
      --out_dir $RES/maskreg_${SEL}_r${R}
  done
done

python dit_inpaint_assemble.py --root $RES --out $RES/table_stage2 --by_bucket
