#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_pipeline.sh — One Verifier -> sparse target execution (DiT).
#
# Stages (roadmap numbering):
#   A (1,3,4)  oracle sparse-execution sweep: suffix_mode x split x hard_ratio,
#              FID-10k vs random-selector sparse at the same budget
#   B (6)      router teacher-dataset collection
#   C (7)      router training (rank regression)
#   D (8)      router standalone eval (AURC / recall / FAR / latency)
#   E (9)      end-to-end learned-router sparse execution + FID
#   F (10)     wall-clock microbenchmark (dense vs sparse, batch sweep)
#
# Run individual stages:  STAGE=A bash run_dit_sparse_pipeline.sh
# Long jobs: run inside tmux. Requires clean-fid for FID stages.
# ============================================================================
set -e
export TMPDIR=${TMPDIR:-/mnt/HDD_12TB/bam_ki/tmp}

TGT=${TGT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt}
DFT=${DFT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt}
REF=${REF:-/mnt/HDD_12TB/bam_ki/datasets/imagenet64/val}
ROOT=${ROOT:-/mnt/HDD_12TB/bam_ki/results/dit_in64_sparse}
RTR=${RTR:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/router_nano.pt}
NFID=${NFID:-10000}          # FID sample count (use 1000 for quick pilots)
STAGE=${STAGE:-ALL}

COMMON="--target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000"

# ---------------------------------------------------------------- Stage A
if [ "$STAGE" = "A" ] || [ "$STAGE" = "ALL" ]; then
  echo "=== Stage A: oracle sparse sweep (suffix x split x ratio) ==="
  # pilot geometry first at split 0.5; widen splits at the best ratio after
  for mode in sparse_mlp sparse_attn; do
    for r in 0.1 0.3 0.5 0.7; do
      for sel in oracle random; do
        OUT=$ROOT/sweep/${sel}_${mode}_m0.5_r${r}
        [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
        python dit_sparse_sampler.py $COMMON \
          --selector $sel --suffix_mode $mode --hard_ratio $r --split 0.5 \
          --n_samples $NFID --batch 128 --steps 50 \
          --dump_samples --ref_dir $REF --out_dir $OUT
      done
    done
  done
  # split ablation at r=0.3 (oracle)
  for m in 0.25 0.75; do
    OUT=$ROOT/sweep/oracle_sparse_mlp_m${m}_r0.3
    [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
    python dit_sparse_sampler.py $COMMON \
      --selector oracle --suffix_mode sparse_mlp --hard_ratio 0.3 --split $m \
      --n_samples $NFID --batch 128 --steps 50 \
      --dump_samples --ref_dir $REF --out_dir $OUT
  done
  # references
  for sel in dense draft mix; do
    OUT=$ROOT/sweep/ref_${sel}
    [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
    python dit_sparse_sampler.py $COMMON \
      --selector $sel --hard_ratio 0.3 --split 0.5 \
      --n_samples $NFID --batch 128 --steps 50 \
      --dump_samples --ref_dir $REF --out_dir $OUT
  done
  python dit_sparse_assemble.py --root $ROOT/sweep
fi

# ---------------------------------------------------------------- Stage B
if [ "$STAGE" = "B" ] || [ "$STAGE" = "ALL" ]; then
  echo "=== Stage B: router teacher dataset ==="
  python dit_router_dataset.py $COMMON \
    --n_traj 2000 --steps 50 --t_stride 2 --batch 64 \
    --seed 0 --out_dir $ROOT/router_data
fi

# ---------------------------------------------------------------- Stage C
if [ "$STAGE" = "C" ] || [ "$STAGE" = "ALL" ]; then
  echo "=== Stage C: router training (rank regression) ==="
  python -m training.train_router \
    --data_dir $ROOT/router_data --out $RTR \
    --loss rank --steps 20000 --batch 64
fi

# ---------------------------------------------------------------- Stage D
if [ "$STAGE" = "D" ] || [ "$STAGE" = "ALL" ]; then
  echo "=== Stage D: router standalone eval ==="
  python dit_router_eval.py $COMMON --router $RTR \
    --n_traj 64 --steps 50 --seed 1234 --out_dir $ROOT/router_eval
fi

# ---------------------------------------------------------------- Stage E
if [ "$STAGE" = "E" ] || [ "$STAGE" = "ALL" ]; then
  echo "=== Stage E: end-to-end learned-router sparse execution ==="
  for mode in sparse_mlp sparse_attn; do
    for r in 0.3 0.5; do
      OUT=$ROOT/sweep/router_${mode}_m0.5_r${r}
      [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
      python dit_sparse_sampler.py $COMMON --router $RTR \
        --selector router --suffix_mode $mode --hard_ratio $r --split 0.5 \
        --n_samples $NFID --batch 128 --steps 50 \
        --dump_samples --ref_dir $REF --out_dir $OUT
    done
  done
  python dit_sparse_assemble.py --root $ROOT/sweep
fi

# ---------------------------------------------------------------- Stage F
if [ "$STAGE" = "F" ] || [ "$STAGE" = "ALL" ]; then
  echo "=== Stage F: wall-clock microbenchmark ==="
  python dit_sparse_bench.py $COMMON \
    --batches 1,4,8,16 --ratios 0.1,0.3,0.5,0.7 --splits 0.25,0.5,0.75 \
    --dtype bf16 --iters 100 --warmup 20 \
    --out $ROOT/sparse_bench_bf16.json
fi

echo "=== pipeline stage(s) $STAGE done ==="
