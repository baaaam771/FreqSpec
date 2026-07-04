#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA7.sh — draft-free token-wise step allocation.
#
# Structural pivot after A5: every prior variant put DRAFT eps on easy tokens,
# so the quality ceiling was draft-bound (Nano draft-only FID 92) and lost to
# step reduction. But A5 itself proved the target's own eps is temporally
# smooth (flat dense curve; anchor hardness persists). So:
#
#   easy tokens  -> REUSE the target's own eps from the last anchor
#   hard tokens  -> recompute via cache_attn (unchanged machinery)
#   hardness     -> anchor-to-anchor eps change  (--selector delta, NO draft)
#
# This reframes the method as token-wise step allocation: dense s(N/cp) is the
# "skip everything between anchors" special case; we additionally refine the
# hard r fraction at the in-between steps. It is the only variant whose
# easy-token quality inherits step-reduction smoothness instead of draft
# quality — the one structure that can land BELOW the dense curve.
#
# Comparison logic: a run at S steps with cache_period cp has S/cp anchors,
# so its dense reference is s(S/cp) PLUS our hard-token work. The question at
# each point: does hard-token refinement between anchors buy more FID than
# spending the same MACs on more dense steps?
#
# Success bar: any delta/anchor point strictly below the dense s10-s25 curve
# in (TOTAL, FID). Diagnostic: oracle upper bound at two key points.
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

runr () {  # sel steps r cp   (m=0, easy_source=target_cache)
  OUT=$ROOT/sweep/${1}TC_cache_attn_s$2_m0.0_r$3_cp$4
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $BASE --steps $2 \
    --selector $1 --easy_source target_cache --suffix_mode cache_attn \
    --hard_ratio $3 --split 0.0 --cache_period $4 --out_dir $OUT
}

# ---- axis 1: s30 cp2 -> 15 anchors == dense s15 grid density
#      (dense s15 = 44.1 @ 0.30; these points add ~0.10-0.14 TOTAL of hard
#       work — success means FID meaningfully below 44 at TOTAL ~0.40-0.45)
for r in 0.1 0.3 0.5; do
  runr delta 30 $r 2
done

# ---- axis 2: s20 cp2 -> 10 anchors == dense s10 grid (48.8 @ 0.20)
#      contested low-compute region: TOTAL ~0.27-0.33
for r in 0.1 0.3 0.5; do
  runr delta 20 $r 2
done

# ---- axis 3: s50 cp5 -> 10 anchors, finer in-between refinement grid
for r in 0.3 0.5; do
  runr delta 50 $r 5
done
runr delta 50 0.3 3

# ---- axis 4: selector comparison at two key points
runr anchor 30 0.3 2       # draft-informed selection (draft at anchors only)
runr oracle 30 0.3 2       # upper bound (diagnostic; oracle cost excluded)
runr random 30 0.3 2       # selection must still matter without the draft
runr oracle 20 0.3 2

python dit_sparse_assemble.py --root $ROOT/sweep
python dit_sparse_pareto.py --root $ROOT/sweep --out $ROOT/figs
echo "=== Stage A7 done ==="
