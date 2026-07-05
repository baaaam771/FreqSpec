#!/usr/bin/env bash
# ============================================================================
# run_dit_inpaint_stage2.sh — token-selective correction on DiT inpainting.
#
# Stage 1 (dense/reduced-step inpainting baselines) is REUSED, not rerun,
# provided its invariants match this pipeline: checkpoint, image list,
# mask_kind/mask_seed (masks here regenerate deterministically from
# (mask_seed, image_index)), scheduler/steps, CFG, per-step known-region
# reinjection  z <- M*z + (1-M)*q_sample(x_known, t), and sampling seed.
# If Stage 1 used a DIFFERENT reinjection, rerun its dense rows through this
# script (selector=dense) so all comparisons share one implementation --
# the in-script dense/r0 controls below cover that case at the main step
# count either way.
#
# Feedback items wired in dit_inpaint_sparse.py:
#   1. frequency selector on the anchor's predicted x0 (not z_t)
#   2. --budget_scope {global, mask} separation
#   3. exact per-image mask-only budget (k_i = round(r * |mask tokens_i|))
#   4. combo = rank-normalized w_d*delta + w_f*freq + w_b*boundary
#   5. boundary / frequency / delta / combo / random / rotate / oracle
#      ablation at matched budgets
#
# Two-factor prediction under test: the hole is where temporal change is
# BOTH concentrated and consequential, so here -- unlike class-conditional
# IN-64 -- informed selection should finally separate from random.
# ============================================================================
set -e
export TMPDIR=${TMPDIR:-/mnt/HDD_12TB/bam_ki/tmp}

TGT=${TGT:-/mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt}
DATA=${DATA:-/mnt/HDD_12TB/bam_ki/datasets/imagenet64/val}
LIST=${LIST:-/mnt/HDD_12TB/bam_ki/datasets/imagenet64/val_list.txt}
LABELS=${LABELS:-}                     # optional: one class id per line
ROOT=${ROOT:-/mnt/HDD_12TB/bam_ki/results/dit_in64_inpaint}
N=${N:-500}
STEPS=${STEPS:-30}
CP=${CP:-2}
MSEED=${MSEED:-0}

BASE="--target $TGT --target_model DiT-S \
      --data_dir $DATA --list_file $LIST \
      --img_size 64 --patch 4 --num_classes 1000 \
      --mask_kind mixed --mask_seed $MSEED \
      --steps $STEPS --cache_period $CP --split 0.0 \
      --n_samples $N --batch 64"
[ -n "$LABELS" ] && BASE="$BASE --labels_file $LABELS"

run () {  # selector scope ratio [extra] [tag]
  OUT=$ROOT/stage2/$1_$2_r$3$5
  [ -f "$OUT/inpaint_summary.json" ] && { echo "skip $OUT"; return; }
  python dit_inpaint_sparse.py $BASE \
    --selector $1 --budget_scope $2 --hard_ratio $3 $4 --out_dir $OUT
  python dit_inpaint_eval.py --run_dir $OUT --data_dir $DATA \
    --out $OUT/eval.json
}

# ---- controls through the SAME pipeline (reinjection-identical)
run dense global 0.3
run r0    mask   0

# ---- selector ablation at exact mask-only budget (items 2,3,5)
for sel in delta frequency boundary random rotate oracle; do
  run $sel mask 0.3
done
run combo mask 0.3

# ---- combo component ablation at the same budget (item 4/5)
run combo mask 0.3 "--w_freq 0 --w_boundary 0" _dOnly
run combo mask 0.3 "--w_delta 0 --w_boundary 0" _fOnly
run combo mask 0.3 "--w_delta 0 --w_freq 0"     _bOnly

# ---- budget sweep on the best-expected selector
for r in 0.1 0.5 0.7; do
  run delta mask $r
done

# ---- scope comparison: same selector, global budget matched in ABSOLUTE k
#      (mask fraction ~0.2 of tokens => global r ~ 0.3*0.2; adjust after
#       checking mean_true_k in the mask-scope summaries)
run delta global 0.06

echo "=== Stage 2 done; summaries: ==="
for f in $ROOT/stage2/*/eval.json; do
  d=$(dirname $f)
  python - "$d" <<'PY'
import json, sys, os
d = sys.argv[1]
e = json.load(open(os.path.join(d, "eval.json")))
s = json.load(open(os.path.join(d, "inpaint_summary.json")))
o = e["overall"]
print(f"{os.path.basename(d):28s} TOTAL={s['total_vs_dense_same_steps']:<7} "
      f"maskLPIPS={o['mask_lpips']:<7} bLPIPS={o['boundary_lpips']:<7} "
      f"kPSNR={o['known_psnr']}")
PY
done
