#!/bin/bash
# ============================================================================
# Stage 1 — DiT inpainting baseline: train target + draft, quality-gate,
#           dense step-reduction sweep (mask-restricted metrics).
#           Run inside tmux on the server. Adjust ROOT paths.
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp

ROOT=/mnt/HDD_12TB/bam_ki
DATA_TR=$ROOT/datasets/imagenet64/train
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp
mkdir -p $CKPT $RES

# 1) target DiT-S-Inp (300k steps, ~ same budget as the generation DiT-S)
python -m training.train_dit_inpaint --model DiT-S-Inp \
    --data_root $DATA_TR --dataset imagenet \
    --out $CKPT/target.pt --steps 300000 --batch 256 --workers 0

# 2) draft DiT-Nano-Inp, region-aware distilled
python -m training.train_dit_inpaint --model DiT-Nano-Inp \
    --data_root $DATA_TR --dataset imagenet \
    --distill_from $CKPT/target.pt --target_model DiT-S-Inp \
    --out $CKPT/draft_nano.pt --steps 300000 --batch 256 --workers 0

# 3) dense step-reduction sweep (the honest frontier, mask metrics)
for S in 50 40 30 25 20 15 10; do
  python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dense --steps $S \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/dense_s$S
done

# 4) draft-only reference (draft-bound floor)
python dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --draft $CKPT/draft_nano.pt --draft_model DiT-Nano-Inp \
    --data_root $DATA_VA --mode draft --steps 50 \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/draft_only

python dit_inpaint_assemble.py --root $RES --out $RES/table_stage1
