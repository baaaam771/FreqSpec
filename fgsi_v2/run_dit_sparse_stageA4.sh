#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA4.sh — Pareto completion after the A3 cache_attn win.
#
# A3 established: oracle cache_attn (cp=5) beats plain sparse_attn everywhere,
# m=0.25 r=0.3 matches the mix reference at 65 percent of dense compute
# (61.8 @ 0.679 vs 61.6 @ 1.045), and the cp knob trades staleness for cost
# monotonically. Three gaps remain before the story is publishable:
#
#  1. HONEST BASELINE — reduced-step dense targets at matched compute
#     (s30/s35/s40/s45). Without these the Pareto plot cannot be placed.
#  2. DEEPER CACHE — m=0.25 rescued means depth-correct context works; push
#     to m=0.25 at higher budgets and m=0 (whole network cached, MAC ~0.56).
#  3. ANCHOR SELECTOR — training-free alternative to the router: reuse the
#     exact d_i measured at the last dense anchor (hard tokens persist across
#     nearby steps). If anchor ~= oracle, the router bar rises; if not, the
#     temporal decay of token hardness is itself a finding the router exploits.
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

# ------------------------------------------------ 1. reduced-step dense refs
for st in 30 35 40 45; do
  OUT=$ROOT/sweep/ref_dense_s${st}
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; continue; }
  python dit_sparse_sampler.py $BASE --steps $st \
    --selector dense --out_dir $OUT
done

run () {  # sel r split cp
  OUT=$ROOT/sweep/$1_cache_attn_m$3_r$2_cp$4_w1.0
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $BASE --steps 50 \
    --selector $1 --suffix_mode cache_attn --hard_ratio $2 --split $3 \
    --cache_period $4 --out_dir $OUT
}

# ------------------------------------------------ 2. deeper-cache frontier
run oracle 0.5 0.25 5      # deep suffix, mid budget
run oracle 0.7 0.25 5      # deep suffix, high budget
run oracle 0.3 0.0  5      # whole-network cache (MAC ~0.56)
run oracle 0.5 0.0  5
run oracle 0.3 0.25 3      # best-efficiency x tighter anchors
run oracle 0.5 0.25 3

# random controls at the new frontier points
run random 0.3 0.25 5
run random 0.3 0.0  5

# ------------------------------------------------ 3. anchor-carried selector
run anchor 0.3 0.5  5      # direct compare vs oracle 60.6 / random 76.0
run anchor 0.5 0.5  5
run anchor 0.3 0.25 5      # at the efficiency point
run anchor 0.3 0.25 3      # fresher anchors help the selector too

python dit_sparse_assemble.py --root $ROOT/sweep
echo "=== Stage A4 done ==="
