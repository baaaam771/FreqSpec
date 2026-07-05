#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA8.sh — paper-hardening controls (reviewer-proofing).
#
#  1. 3-SEED EQUIVALENCE at the inertness point (s30, m=0, r=0.3, cp=2,
#     target-cache): oracle vs random vs anchor. The single-seed ordering
#     (random 44.31 < oracle 44.53) is within seed sigma~0.2; the paper's
#     "selection is inert" claim needs the CI. NOTE: oracleTC already uses
#     the correct cache-oracle score ||eps_t^tgt - eps_anchor^tgt|| (current
#     vs cached target), so equivalence here is the real finding.
#  2. DENSE MICRO-FRONTIER s11-s14: fills TOTAL 0.22-0.28 where the s20
#     reuse points live; needed to judge domination honestly.
#  3. r=0 PURE TEMPORAL CACHING (most important control): all tokens reuse
#     the anchor target eps, zero correction. If r=0 ~= r=0.3, correction
#     itself contributes ~nothing and caching carries the quality.
#     (r=0 at cp=2 on an s-step grid = eps reuse on a 2x-fine t-grid;
#      its gap to dense s/2 isolates the value of the finer grid.)
#  4. ROTATE schedule: round-robin coverage (every token refreshed within
#     1/r sparse steps) -- the standard non-score alternative from the
#     caching literature; one row against score-based selection.
#  5. HETEROGENEITY v2: adds absolute/relative magnitude (the missing panel
#     of the two-factor argument) and 2x2/4x4 block-capture (how much of the
#     token-level change mass block-level top-k preserves -> block-sparse
#     feasibility number for the discussion).
# ============================================================================
set -e
export TMPDIR=${TMPDIR:-/mnt/HDD_12TB/bam_ki/tmp}

TGT=${TGT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt}
DFT=${DFT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt}
REF=${REF:-/mnt/HDD_12TB/bam_ki/datasets/imagenet64/val}
ROOT=${ROOT:-/mnt/HDD_12TB/bam_ki/results/dit_in64_sparse}
NFID=${NFID:-10000}

BASE="--target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
      --img_size 64 --patch 4 --num_classes 1000 \
      --n_samples $NFID --batch 128 --dump_samples --ref_dir $REF"

# ---------------------------------------- 1. 3-seed equivalence (inertness)
for sd in 0 1 2; do
  for sel in oracle random anchor; do
    OUT=$ROOT/sweep/seed${sd}_${sel}TC_cache_attn_s30_m0.0_r0.3_cp2
    [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
    python dit_sparse_sampler.py $BASE --seed $sd --steps 30 \
      --selector $sel --easy_source target_cache --suffix_mode cache_attn \
      --hard_ratio 0.3 --split 0.0 --cache_period 2 --out_dir $OUT
  done
done

# ---------------------------------------- 2. dense micro-frontier
for st in 11 12 13 14; do
  OUT=$ROOT/sweep/ref_dense_s${st}
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
  python dit_sparse_sampler.py $BASE --steps $st --selector dense --out_dir $OUT
done

# ---------------------------------------- 3. r=0 pure temporal caching
r0 () {  # steps cp
  OUT=$ROOT/sweep/r0TC_cache_attn_s$1_m0.0_r0_cp$2
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $BASE --steps $1 \
    --selector random --easy_source target_cache --suffix_mode cache_attn \
    --hard_ratio 0 --split 0.0 --cache_period $2 --out_dir $OUT
}
r0 20 2
r0 30 2
r0 50 3
r0 50 5

# ---------------------------------------- 4. rotate schedule (one row)
for sd in 0 1 2; do
  OUT=$ROOT/sweep/seed${sd}_rotateTC_cache_attn_s30_m0.0_r0.3_cp2
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
  python dit_sparse_sampler.py $BASE --seed $sd --steps 30 \
    --selector rotate --easy_source target_cache --suffix_mode cache_attn \
    --hard_ratio 0.3 --split 0.0 --cache_period 2 --out_dir $OUT
done

# ---------------------------------------- 5. heterogeneity v2
python dit_heterogeneity.py \
  --target $TGT --target_model DiT-S \
  --img_size 64 --patch 4 --num_classes 1000 \
  --n_traj 64 --steps 50 \
  --out $ROOT/heterogeneity_v2

python dit_sparse_assemble.py --root $ROOT/sweep
echo "=== Stage A8 done ==="
