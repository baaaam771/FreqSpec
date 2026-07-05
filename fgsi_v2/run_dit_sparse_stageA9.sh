#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA9.sh — hardening the r=0 result (the paper's one
# positive finding). Four axes, in priority order:
#
#  1. 3-SEED r=0 at all four winning points. The s30/cp2 margin over dense
#     s15 is only 0.44 FID -- a CI is near-mandatory; the 0.20x points
#     (margin ~3 FID) are repeated for uniform reporting.
#  2. GRID-DENSITY SWEEP at a FIXED budget of 10 target evaluations:
#     S in {10,20,30,50,100} with cp=S/10. S=10/cp=1 must reproduce dense
#     s10 exactly (built-in sanity); a monotone FID improvement with grid
#     density is the direct test of the finer-grid-integration reading.
#  3. END-TO-END WALL-CLOCK: dense s10/s15 vs the r=0 configs under matched
#     conditions (same batch, bf16 autocast, warm-up excluded, CUDA sync,
#     no image saving). r=0 does 2-5x more scheduler updates at equal
#     target evals; this measures whether the FID win survives in seconds.
#  4. SAME-SEED PAIRED COMPARISON: dense s10 vs r=0 s50/cp5 and dense s15
#     vs r=0 s30/cp2, both against the same-seed dense s50 reference --
#     per-image LPIPS/MSE, paired win rate, sign test. Removes the
#     distributional variance FID pools.
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

r0run () {  # steps cp seed
  OUT=$ROOT/sweep/seed$3_r0TC_cache_attn_s$1_m0.0_r0_cp$2
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $BASE --seed $3 --steps $1 \
    --selector random --easy_source target_cache --suffix_mode cache_attn \
    --hard_ratio 0 --split 0.0 --cache_period $2 --out_dir $OUT
}

# --------------------------------- 1. r=0 3-seed (seed 0 exists from A8)
for sd in 1 2; do
  r0run 20 2 $sd
  r0run 30 2 $sd
  r0run 50 5 $sd
  r0run 50 3 $sd
done

# --------------------------------- 2. grid-density sweep @ 10 target evals
# S=10/cp=1 == dense s10 (sanity); S=20/cp=2 and S=50/cp=5 exist at seed 0
r0run 10 1 0
r0run 30 3 0
r0run 100 10 0

# --------------------------------- 3. end-to-end wall-clock
python dit_e2e_bench.py \
  --target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
  --img_size 64 --patch 4 --num_classes 1000 \
  --batch 128 --repeats 7 --dtype bf16 \
  --out $ROOT/e2e_wallclock_b128_bf16.json
python dit_e2e_bench.py \
  --target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
  --img_size 64 --patch 4 --num_classes 1000 \
  --batch 1 --repeats 15 --dtype bf16 \
  --out $ROOT/e2e_wallclock_b1_bf16.json

# --------------------------------- 4. same-seed paired comparison
python dit_paired_compare.py \
  --target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
  --img_size 64 --patch 4 --num_classes 1000 \
  --pair 10,50,5 --pair 15,30,2 \
  --ref_steps 50 --n_samples 512 --batch 128 \
  --out $ROOT/paired_compare

python dit_sparse_assemble.py --root $ROOT/sweep
echo "=== Stage A9 done ==="
