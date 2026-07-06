#!/bin/bash
# ============================================================================
# Stage 1 — DiT inpainting baseline. RESUMABLE.
#   - 학습은 out.pt / out.pt.last.pt 를 항상 덮어써서 어디서 죽어도 자동 resume.
#   - milestone(--milestone_every)로 영구 snapshot(out.pt.stepN.pt)도 남김.
#   - dense/draft sweep는 metrics.json 있으면 자동 skip → 스크립트 재실행 시 이어감.
#   tmux 권장:  tmux new -s stage1 'bash run_dit_inpaint_stage1.sh 2>&1 | tee stage1.log'
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp

ROOT=/mnt/HDD_12TB/bam_ki
DATA_TR=$ROOT/datasets/imagenet64/train
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp
mkdir -p $CKPT $RES

# 1) target DiT-S-Inp — 이미 target.pt 있으면 자동 resume(step 이어감).
#    완료됐으면 "already at/after target steps" 출력 후 즉시 통과.
python -u -m training.train_dit_inpaint --model DiT-S-Inp \
    --data_root $DATA_TR --dataset imagenet \
    --out $CKPT/target.pt --steps 300000 --batch 256 --workers 8 \
    --save_every 5000 --milestone_every 50000

# 2) draft DiT-Nano-Inp, region-aware distilled — draft_nano.pt 없으면 새로,
#    있으면 이어서. (지금 상황: target.pt만 있으니 여기서부터 실제로 학습됨)
python -u -m training.train_dit_inpaint --model DiT-Nano-Inp \
    --data_root $DATA_TR --dataset imagenet \
    --distill_from $CKPT/target.pt --target_model DiT-S-Inp \
    --out $CKPT/draft_nano.pt --steps 300000 --batch 256 --workers 8 \
    --save_every 5000 --milestone_every 50000

# 3) dense step-reduction sweep (완료된 out_dir은 자동 skip)
for S in 50 40 30 25 20 15 10; do
  python -u dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --data_root $DATA_VA --mode dense --steps $S \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/dense_s$S
done

# 4) draft-only floor
python -u dit_inpaint_sampler.py \
    --target $CKPT/target.pt --target_model DiT-S-Inp \
    --draft $CKPT/draft_nano.pt --draft_model DiT-Nano-Inp \
    --data_root $DATA_VA --mode draft --steps 50 \
    --n_samples 200 --batch 32 --ref_dir $RES/ref \
    --out_dir $RES/draft_only

python -u dit_inpaint_assemble.py --root $RES --out $RES/table_stage1
