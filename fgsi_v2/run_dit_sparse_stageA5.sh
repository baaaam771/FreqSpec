#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA5.sh — sparse x step-reduction combination.
#
# A4 verdict:
#   * anchor selector ~= oracle everywhere (within 1-2 FID; random +15) ->
#     token hardness is temporally persistent; the learned router is
#     deprioritized to an ablation. Anchor is the deployable selector and its
#     d_i measurement is free at the anchors cache_attn already needs.
#   * reduced-step dense DOMINATES the 50-step sparse frontier: s30 = FID 42.4
#     at TOTAL 0.60, below every sparse point. The DDIM curve is flat 30-50.
#
# Strategy: step reduction and token sparsity are ORTHOGONAL axes. Step
# reduction is free until its knee (typically ~25 steps for DDIM); sparsity
# cuts per-step cost independently. So the contested region is TOTAL compute
# BELOW the knee: cache_attn on a 25-30 step schedule reaches TOTAL 0.30-0.45,
# where the dense equivalent (s15-s22) is expected to degrade.
#
# This script (1) maps the dense knee (s10-s25), (2) runs anchor-selected
# cache_attn on short schedules with cp scaled so the anchor Delta-t span
# matches the s50 sweeps (s50 cp5 ~= s30 cp3 ~= s25 cp2-3).
#
# Success bar: any combined point strictly below the dense step-reduction
# curve in (TOTAL, FID). Read results with dit_sparse_assemble.py's new
# TOTAL column (compute relative to 50-step dense).
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

# --------------------------------------------- 1. dense knee mapping
for st in 10 15 20 25; do
  OUT=$ROOT/sweep/ref_dense_s${st}
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
  python dit_sparse_sampler.py $BASE --steps $st --selector dense --out_dir $OUT
done

runc () {  # steps r split cp   (anchor selector, cache_attn)
  OUT=$ROOT/sweep/anchor_cache_attn_s$1_m$3_r$2_cp$4
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $BASE --steps $1 \
    --selector anchor --suffix_mode cache_attn --hard_ratio $2 --split $3 \
    --cache_period $4 --out_dir $OUT
}

# --------------------------------------------- 2. sparse on short schedules
# s30: matched anchor Delta-t -> cp=3; also cp=2 (fresher) and cp=5 (cheaper)
for r in 0.3 0.5; do
  for m in 0.0 0.25; do
    runc 30 $r $m 3
  done
done
runc 30 0.3 0.0 2
runc 30 0.3 0.0 5
runc 30 0.5 0.25 2

# s25: knee edge; cp=2-3
for r in 0.3 0.5; do
  runc 25 $r 0.0 2
  runc 25 $r 0.25 3
done

# oracle upper bounds at two key combined points (diagnostic)
for cfg in "30 0.3 0.0 3" "25 0.3 0.0 2"; do
  set -- $cfg
  OUT=$ROOT/sweep/oracle_cache_attn_s$1_m$3_r$2_cp$4
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
  python dit_sparse_sampler.py $BASE --steps $1 \
    --selector oracle --suffix_mode cache_attn --hard_ratio $2 --split $3 \
    --cache_period $4 --out_dir $OUT
done

python dit_sparse_assemble.py --root $ROOT/sweep
echo "=== Stage A5 done ==="
