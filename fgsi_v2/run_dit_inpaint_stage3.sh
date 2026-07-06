#!/bin/bash
# ============================================================================
# Stage 3 (REVISED v2) — global vs WITHIN-MASK selectors, frequency source,
# and the pure-frequency-contribution ablation.
#
# 리뷰 반영:
#  - region=mask 는 이제 STRICT hole (boundary 누출 없음). boundary를 포함하려면
#    region=mask_plus_boundary 사용.
#  - mask-restricted 실험 이름을 within_mask_* 로 정리 (mask eligibility 안에서의
#    ranking 실험임을 명확히).
#  - 핵심 비교: within_mask_boundary_delta  vs  within_mask_full_combo
#    (frequency 순수 기여).
# ============================================================================
set -e
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
ROOT=/mnt/HDD_12TB/bam_ki
DATA_VA=$ROOT/datasets/imagenet64/val
CKPT=$ROOT/ckpt_dit_inp
RES=$ROOT/results/dit_inp
REF=$RES/ref
COMMON="--target $CKPT/target.pt --target_model DiT-S-Inp --data_root $DATA_VA \
        --mode dace --suffix cache --easy anchor --steps 30 --cache_period 2 \
        --split_m 0 --hard_ratio 0.3 --n_samples 200 --batch 32 --ref_dir $REF"

# ---- GLOBAL selectors (전체 token 점수; group A) ----
for SEL in random freq delta oracle; do
  python -u dit_inpaint_sampler.py $COMMON --region global --budget ratio \
    --selector $SEL --freq_src x0_anchor \
    --out_dir $RES/global_${SEL}
done

# ---- frequency SOURCE ablation (item 1): zt vs x0_anchor vs known ----
for FS in zt x0_anchor known; do
  python -u dit_inpaint_sampler.py $COMMON --region global --budget ratio \
    --selector freq --freq_src $FS \
    --out_dir $RES/global_freq_${FS}
done

# ---- WITHIN-MASK ranking ablation (item 13) ----
# strict mask eligibility; combo 가중치로 ranking 신호만 바꿈.
# within_mask_random : mask 안에서 무작위 (선택 자체 하한)
python -u dit_inpaint_sampler.py $COMMON --region mask --budget ratio \
  --selector random --out_dir $RES/within_mask_random
# within_mask_only : mask/boundary tie-break (mask 항은 상수라 사실상 tie-break)
python -u dit_inpaint_sampler.py $COMMON --region mask --budget ratio \
  --selector mask --out_dir $RES/within_mask_only
# within_mask_frequency : mask 안에서 frequency ranking
python -u dit_inpaint_sampler.py $COMMON --region mask --budget ratio \
  --selector combo --cw_mask 0 --cw_bnd 0 --cw_freq 1 --cw_delta 0 \
  --freq_src x0_anchor --out_dir $RES/within_mask_frequency
# within_mask_delta : mask 안에서 delta ranking
python -u dit_inpaint_sampler.py $COMMON --region mask --budget ratio \
  --selector combo --cw_mask 0 --cw_bnd 0 --cw_freq 0 --cw_delta 1 \
  --out_dir $RES/within_mask_delta
# within_mask_boundary_delta : frequency 없는 강한 기준 (mask+boundary eligible)
python -u dit_inpaint_sampler.py $COMMON --region mask_plus_boundary --budget ratio \
  --selection_boundary_k 2 --selector combo --cw_mask 0 --cw_bnd 1 --cw_freq 0 --cw_delta 1 \
  --out_dir $RES/within_mask_boundary_delta
# within_mask_full_combo : frequency 추가 (핵심 비교의 다른 축)
python -u dit_inpaint_sampler.py $COMMON --region mask_plus_boundary --budget ratio \
  --selection_boundary_k 2 --selector combo --cw_mask 0 --cw_bnd 1 --cw_freq 1 --cw_delta 1 \
  --freq_src x0_anchor --out_dir $RES/within_mask_full_combo
# within_mask_oracle : mask 안 상한
python -u dit_inpaint_sampler.py $COMMON --region mask --budget ratio \
  --selector oracle --out_dir $RES/within_mask_oracle

python -u dit_inpaint_assemble.py --root $RES --out $RES/table_stage3 --by_bucket
