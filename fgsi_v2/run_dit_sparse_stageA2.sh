#!/usr/bin/env bash
# ============================================================================
# run_dit_sparse_stageA2.sh — staleness-control sweep after Stage-A findings.
#
# Stage-A pilot showed:
#   * oracle >> random at every budget (gate PASSED -> router justified)
#   * sparse_attn dominates sparse_mlp on BOTH axes (frozen-but-valid easy
#     context beats attention-without-MLP OOD states) -> sparse_attn only here
#   * damage grows with sparse-suffix depth (m=0.25 catastrophic) ->
#     bound staleness instead of shrinking the suffix:
#       --refresh_every 2   dense refresh of every 2nd suffix block
#       --dense_until 0.7   dense-target warm-up while t/T > 0.7
#
# Success bar: oracle sparse_attn + refresh + warm-up should approach the
# output-mixing reference (mix r=0.3, FID ~61.6) at clearly LOWER total MACs,
# and approach dense (42.3) as r -> 0.7.
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

run () {  # sel mode r split refresh warm
  OUT=$ROOT/sweep/$1_$2_m$4_r$3_re$5_w$6
  [ -f "$OUT/sparse_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_sparse_sampler.py $COMMON \
    --selector $1 --suffix_mode $2 --hard_ratio $3 --split $4 \
    --refresh_every $5 --dense_until $6 --out_dir $OUT
}

# axis 1: refresh alone (isolate the staleness fix)
for r in 0.3 0.5 0.7; do
  run oracle sparse_attn $r 0.5 2 1.0
done

# axis 2: refresh + warm-up (full staleness control)
for r in 0.3 0.5 0.7; do
  run oracle sparse_attn $r 0.5 2 0.7
done

# axis 3: warm-up alone at the mid budget (decompose the two fixes)
run oracle sparse_attn 0.5 0.5 0 0.7

# axis 4: refresh period ablation at the mid budget
run oracle sparse_attn 0.5 0.5 3 0.7

# random controls at the full-control setting (selection must still matter)
for r in 0.3 0.5; do
  run random sparse_attn $r 0.5 2 0.7
done

python dit_sparse_assemble.py --root $ROOT/sweep
echo "=== Stage A2 done ==="
