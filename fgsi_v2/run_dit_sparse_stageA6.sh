#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA6.sh — paper-closing experiments. NO more sweeps.
#
# A5 verdict: the dense DDIM curve has no exploitable knee on this DiT-S
# (s10 = 48.8 @ 0.20); dense step reduction beats every spatial point,
# INCLUDING the mixing ceiling. Final framing therefore pivots from
# "compute Pareto win" to the mechanism + ceiling story:
#
#   C1  first actual sparse execution of verifier decisions
#   C2  depth-frozen context fails (OOD) -> depth-correct temporal cache
#       fully recovers it (154.8 -> 61.8 at matched budget), with exactness
#       under a fresh cache as the theoretical anchor
#   C3  anchor-carried selection ~= oracle (temporal persistence of hardness)
#   C4  ceiling analysis: cached execution attains the mixing ceiling at
#       0.56-0.68x compute; the ceiling itself losing to step reduction is a
#       property of draft/target scale, already conceded by One Verifier
#
# This script fills the three holes that framing still has:
#   1. mix ceilings at r=0.5, 0.7 (s50) and r=0.3 (s30) — closure must be
#      shown at every budget and on a short schedule, not just r=0.3/s50
#   2. seed variance (3 seeds) on the headline configs
#   3. wall-clock bench (Stage F) for the honest latency table
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

# --------------------------------- 1. ceiling closure across budgets/schedules
for r in 0.5 0.7; do
  OUT=$ROOT/sweep/mix_r${r}_s50
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
  python dit_sparse_sampler.py $BASE --steps 50 \
    --selector mix --hard_ratio $r --out_dir $OUT
done
OUT=$ROOT/sweep/mix_r0.3_s30
[ -f "$OUT/sparse_summary.json" ] || python dit_sparse_sampler.py $BASE \
  --steps 30 --selector mix --hard_ratio 0.3 --out_dir $OUT

# cache points needed to pair with the new ceilings (fill any missing)
OUT=$ROOT/sweep/anchor_cache_attn_s50_m0.25_r0.7_cp5
[ -f "$OUT/sparse_summary.json" ] || python dit_sparse_sampler.py $BASE \
  --steps 50 --selector anchor --suffix_mode cache_attn --hard_ratio 0.7 \
  --split 0.25 --cache_period 5 --out_dir $OUT
OUT=$ROOT/sweep/anchor_cache_attn_s50_m0.25_r0.5_cp5
[ -f "$OUT/sparse_summary.json" ] || python dit_sparse_sampler.py $BASE \
  --steps 50 --selector anchor --suffix_mode cache_attn --hard_ratio 0.5 \
  --split 0.25 --cache_period 5 --out_dir $OUT

# --------------------------------- 2. seed variance on headline configs
for sd in 1 2; do   # seed 0 already exists from earlier stages
  OUT=$ROOT/sweep/seed${sd}_oracle_cache_attn_s50_m0.25_r0.3_cp5
  [ -f "$OUT/sparse_summary.json" ] || python dit_sparse_sampler.py $BASE \
    --seed $sd --steps 50 --selector oracle --suffix_mode cache_attn \
    --hard_ratio 0.3 --split 0.25 --cache_period 5 --out_dir $OUT
  OUT=$ROOT/sweep/seed${sd}_anchor_cache_attn_s50_m0.25_r0.3_cp5
  [ -f "$OUT/sparse_summary.json" ] || python dit_sparse_sampler.py $BASE \
    --seed $sd --steps 50 --selector anchor --suffix_mode cache_attn \
    --hard_ratio 0.3 --split 0.25 --cache_period 5 --out_dir $OUT
  OUT=$ROOT/sweep/seed${sd}_mix_r0.3_s50
  [ -f "$OUT/sparse_summary.json" ] || python dit_sparse_sampler.py $BASE \
    --seed $sd --steps 50 --selector mix --hard_ratio 0.3 --out_dir $OUT
done

# --------------------------------- 3. wall-clock bench (honest latency table)
python dit_sparse_bench.py \
  --target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
  --img_size 64 --patch 4 --num_classes 1000 \
  --batches 1,4,8,16 --ratios 0.1,0.3,0.5,0.7 --splits 0.0,0.25,0.5 \
  --dtype bf16 --iters 100 --warmup 20 \
  --out $ROOT/sparse_bench_bf16.json

python dit_sparse_assemble.py --root $ROOT/sweep
python dit_sparse_pareto.py --root $ROOT/sweep --out $ROOT/figs
echo "=== Stage A6 done — paper data complete ==="
