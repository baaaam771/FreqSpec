#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA3.sh — temporal-cache sparse execution (Stage 13
# mechanism), after Stage-A2 falsified the within-step staleness controls.
#
# A2 finding: refresh_every made every budget slightly WORSE at higher MACs;
# dense warm-up alone gave <3 FID. Revised diagnosis: the failure is DEPTH
# mismatch, not time staleness — frozen easy tokens are depth-m states fed as
# context into deeper blocks that were trained on their predecessors' outputs.
#
# cache_attn fixes exactly this: easy-token context at suffix block j comes
# from the last dense ANCHOR step's block-j input (depth-CORRECT, time-stale
# by <= cache_period steps). Verified property: with a fresh cache the sparse
# pass equals the dense pass exactly at ANY ratio — all error is temporal.
#
# Success bar (vs the mix reference, FID 61.6 @ MAC 1.045):
#   oracle cache_attn should approach mix FID at MAC ~0.80-0.88 and clearly
#   beat plain sparse_attn at matched MAC (e.g. r=0.5 base was 78.7 @ 0.832).
# ============================================================================
set -e
export TMPDIR=${TMPDIR:-/mnt/HDD_12TB/bam_ki/tmp}

TGT=${TGT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt}
DFT=${DFT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt}
REF=${REF:-/mnt/HDD_12TB/bam_ki/datasets/imagenet64/val}
ROOT=${ROOT:-/mnt/HDD_12TB/bam_ki/results/dit_in64_sparse}
NFID=${NFID:-10000}

COMMON="--target $TGT --target_model DiT-S --draft $DFT --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --n_samples $NFID --batch 128 --steps 50 --dump_samples --ref_dir $REF"

run () {  # sel r split cache_period warm
  OUT=$ROOT/sweep/$1_cache_attn_m$3_r$2_cp$4_w$5
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $COMMON \
    --selector $1 --suffix_mode cache_attn --hard_ratio $2 --split $3 \
    --cache_period $4 --dense_until $5 --refresh_every 0 --out_dir $OUT
}

# axis 1: budget sweep at cp=5 (matches the earlier r sweep for direct compare)
for r in 0.1 0.3 0.5 0.7; do
  run oracle $r 0.5 5 1.0
done

# axis 2: cache-period ablation at r=0.3 (staleness-vs-cost curve)
for cp in 3 10; do
  run oracle 0.3 0.5 $cp 1.0
done

# axis 3: deeper sparse suffix now that context is depth-correct
#         (m=0.25 was catastrophic for sparse_attn; cache should rescue it)
run oracle 0.3 0.25 5 1.0

# axis 4: + warm-up on the best expected point
run oracle 0.3 0.5 5 0.7

# random controls (selection must still matter under the cache)
for r in 0.3 0.5; do
  run random $r 0.5 5 1.0
done

python dit_sparse_assemble.py --root $ROOT/sweep
echo "=== Stage A3 done ==="
