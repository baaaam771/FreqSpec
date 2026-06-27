#!/usr/bin/env bash
# DiT accept-budget sweep (no retraining). Runs the K=1 token-mixing sampler at
# four accept ratios with the 32px DiT-Nano draft; each run writes its own
# sampling_summary.json under sweep_r<R>/. Assemble with dit_assemble_budget.py.
set -e
export TMPDIR=${TMPDIR:-/mnt/HDD_12TB/bam_ki/tmp}

TGT=${TGT:-/mnt/HDD_12TB/bam_ki/ckpt_dit/target.pt}
DFT=${DFT:-/mnt/HDD_12TB/bam_ki/ckpt_dit/draft_nano.pt}
ROOT=${ROOT:-/mnt/HDD_12TB/bam_ki/results/dit_token_poc_nano}

for r in 0.3 0.5 0.7 0.9; do
  echo "=== accept_ratio $r ==="
  python dit_token_sampler.py \
    --target "$TGT" --target_model DiT-S \
    --draft  "$DFT" --draft_model DiT-Nano \
    --out_dir "$ROOT/sweep_r$r" \
    --n_samples 64 --steps 50 --accept_ratio "$r"
done
echo "=== sweep done; assemble with dit_assemble_budget.py ==="
