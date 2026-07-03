# One Verifier, Many Regions: Task- and Backbone-Agnostic Speculative Verification for Conditional Diffusion

Code for the AAAI 2027 submission. This repository implements a **task- and
backbone-agnostic speculative verifier** for conditional diffusion that operates
over an arbitrary verification region Ω — a mask (inpainting), a whole field
(super-resolution), or a transformer token grid (DiT) — with a single shared
acceptance rule.

> **Anonymous submission.** This repository is anonymized for double-blind review.
> It contains no author names, institutions, or identifying information.

> **Scope.** We do **not** claim wall-clock acceleration: the target and the draft
> are both evaluated for verification. The contribution is a single verifier
> formulation whose target usage is controllable across tasks and backbones, and an
> analysis of when the frequency prior helps.

---

## What the verifier does

At each denoising step a frozen **target** predicts noise ε and a compact **draft**
predicts ε over the same field. Both are patchified over a verification region Ω.
For each patch/token the shared core computes an ε-agreement score and a
predicted-`x0` disagreement (risk), modulates strictness with a saliency prior, and
returns a soft per-patch accept weight; accepted patches use the draft, rejected
patches fall back to the target. Only Ω and the saliency prior are task-dependent;
the acceptance rule is shared.

- **Task-agnostic** — the shared acceptance rule; Ω and the saliency prior may
  remain task-dependent.
- **Backbone-agnostic** — depends only on predictions in a shared output space
  (ε and predicted `x0`); no architecture-specific intermediate features. The same
  rule applies to a U-Net feature map and a DiT token grid.

Three instances in the paper:

| Instance | Backbone | Ω | Proves |
|---|---|---|---|
| Inpainting | U-Net (SDXL-Inpainting) | mask | spatial selection |
| Super-resolution | U-Net (SD ×4 upscaler) | whole field | task portability |
| DiT token grid | Transformer (DiT) | patch tokens | backbone portability |

---

## Repository layout

```
models/
  dit.py                     # DiT backbone (configs: DiT-B/S/Ti/Nano)
  draft.py                   # compact draft model
  target_wrapper.py          # frozen target (inpainting)
  sr_target_wrapper.py       # frozen SR target (v-pred -> eps wrapper)
  wavelet.py                 # DWT / wavelet saliency
training/
  scheduler.py               # DDPM/DDIM schedule (pixel-space linear beta)
  train_dit.py               # DiT training (CIFAR-10 / ImageFolder / ImageNet)
  train_sr.py                # SR draft training
inference/                   # speculative verifier core + samplers
dit_imagenet_sanity.py       # Phase 0: dataloader / forward sanity
dit_quality_gate.py          # Phase 1: target/draft sample-quality gate
dit_token_analysis.py        # Phase 2: per-timestep AURC (reliability)
dit_token_sampler.py         # Phase 3: K=1 token-mixing sampler (selectors)
dit_token_fid.py             # Phase 4: FID-10k budget sweep / selector / seeds
dit_in64_figure.py           # assemble the ImageNet-64 mixing figure
imagenet_parquet_to_folder.py# convert HF ImageNet-64 parquet -> ImageFolder
```

---

## Environment

- Python 3.11+ (tested on 3.14), PyTorch + torchvision, CUDA GPU.
- Extra packages: `pyarrow` (parquet conversion), `clean-fid` (FID), `matplotlib`,
  `Pillow`, and a wavelet transform used by `models/wavelet.py`.

```bash
pip install torch torchvision pyarrow clean-fid matplotlib pillow
```

> **Note (Python 3.14).** `clean-fid` is invoked with `--workers 0`; the 3.14
> `forkserver` default cannot pickle its internal resizer.

---

## Schedule (important)

All DiT training and sampling use a **pixel-space linear** noise schedule
(`beta_start=1e-4`, `beta_end=0.02`, `linear`), so `alphas_cumprod[-1] ≈ 0`. The
Stable-Diffusion latent default (`scaled_linear`, `0.00085–0.012`) leaves residual
signal at the final step and must **not** be used for pixel-space DiTs — training
and sampling schedules must match, or samples degrade to noise texture.

---

## Reproducing the DiT / ImageNet-64 study

The DiT study runs as staged go/no-go phases. Set paths first:

```bash
export TMPDIR=/path/to/scratch
DATA=/path/to/imagenet64
CKPT=/path/to/ckpt_dit_in64
RES=/path/to/results/dit_in64
```

### Phase 0 — data + sanity

HF ImageNet-64 ships as parquet (`image={bytes,path}`, `label=int`). Convert to an
ImageFolder layout, then run the sanity check.

```bash
python imagenet_parquet_to_folder.py \
  --parquet_glob "$DATA/data/train-*.parquet" --out_root "$DATA/train" --img_size 64
python imagenet_parquet_to_folder.py \
  --parquet_glob "$DATA/data/validation-*.parquet" --out_root "$DATA/val" --img_size 64

python dit_imagenet_sanity.py --data_root "$DATA/train" \
  --img_size 64 --patch 4 --num_classes 1000 --model DiT-S
# expect: 1,281,167 images / 1000 classes, [B,3,64,64], 256 tokens, PASS
```

### Phase 1 — train target + draft, then quality gate

```bash
python -m training.train_dit --model DiT-S    --dataset imagenet \
  --img_size 64 --patch 4 --num_classes 1000 --data_root "$DATA/train" \
  --out "$CKPT/target.pt"     --steps 300000 --batch 128
python -m training.train_dit --model DiT-Nano --dataset imagenet \
  --img_size 64 --patch 4 --num_classes 1000 --data_root "$DATA/train" \
  --out "$CKPT/draft_nano.pt" --steps 300000 --batch 128

python dit_quality_gate.py \
  --target "$CKPT/target.pt" --target_model DiT-S \
  --draft  "$CKPT/draft_nano.pt" --draft_model DiT-Nano \
  --img_size 64 --patch 4 --num_classes 1000 \
  --out_dir "$RES/quality_gate" --n_samples 64 --steps 50
# gate: target recognizable (not collapsed); draft weaker but not pure noise
```

### Phase 2 — reliability (per-timestep AURC)

```bash
python dit_token_analysis.py \
  --target "$CKPT/target.pt" --target_model DiT-S \
  --draft  "$CKPT/draft_nano.pt" --draft_model DiT-Nano \
  --dataset imagenet --img_size 64 --patch 4 --num_classes 1000 \
  --data_root "$DATA/val" --num_batches 40 --accept_ratio 0.7 \
  --out_dir "$RES/reliability"
# expect: eps-agreement == oracle, ~-58% AURC vs random; frequency/token-norm ~ random
```

### Phase 3 — token-mixing budget sweep

```bash
for AR in 0.3 0.5 0.7 0.9; do
  python dit_token_sampler.py \
    --target "$CKPT/target.pt" --target_model DiT-S \
    --draft  "$CKPT/draft_nano.pt" --draft_model DiT-Nano \
    --img_size 64 --patch 4 --num_classes 1000 \
    --n_samples 64 --steps 50 --accept_ratio $AR \
    --out_dir "$RES/mixing_ar${AR}"
done
```

### Phase 4 — FID-10k (budget sweep, selector ablation, seeds)

```bash
# budget sweep
for AR in 0.3 0.5 0.7 0.9; do
  python dit_token_fid.py \
    --target "$CKPT/target.pt" --target_model DiT-S \
    --draft  "$CKPT/draft_nano.pt" --draft_model DiT-Nano \
    --img_size 64 --patch 4 --num_classes 1000 --ref_dir "$DATA/val" \
    --n_samples 10000 --steps 50 --accept_ratio $AR --seed 0 \
    --out_dir "$RES/fid_ar${AR}"
done

# selector ablation (accept 0.7)
python dit_token_fid.py \
  --target "$CKPT/target.pt" --target_model DiT-S \
  --draft  "$CKPT/draft_nano.pt" --draft_model DiT-Nano \
  --img_size 64 --patch 4 --num_classes 1000 --ref_dir "$DATA/val" \
  --n_samples 10000 --steps 50 --accept_ratio 0.7 \
  --methods "freqspec,eps_cosine,frequency,token_norm,random" \
  --out_dir "$RES/fid_selector_ar0.7"

# seed repeat (accept 0.7)
for SEED in 0 1 2; do
  python dit_token_fid.py \
    --target "$CKPT/target.pt" --target_model DiT-S \
    --draft  "$CKPT/draft_nano.pt" --draft_model DiT-Nano \
    --img_size 64 --patch 4 --num_classes 1000 --ref_dir "$DATA/val" \
    --n_samples 10000 --steps 50 --accept_ratio 0.7 --seed $SEED \
    --methods "freqspec,random" --out_dir "$RES/fid_seed${SEED}_ar0.7"
done
```

Assemble the paper's ImageNet-64 mixing figure:

```bash
python dit_in64_figure.py --grid_dir "$RES/mixing_ar0.7" \
  --out figures/in64_mixing.pdf --crop 5 --cell 66 \
  --fid_target 42.49 --fid_draft 93.12 --fid_freqspec 62.06 --fid_random 79.84
```

---

## Selectors

`dit_token_sampler.py` / `dit_token_fid.py` expose the same selectors compared in
the reliability study, via `--methods`:

| selector | signal |
|---|---|
| `freqspec` (ε-L2) | prediction agreement (default) |
| `eps_cosine` | direction-only agreement |
| `frequency` | wavelet high-frequency saliency |
| `token_norm` | target prediction magnitude |
| `random` | random selection at matched budget |
| `target` / `draft` | quality upper / lower bound |

---

## Key results (ImageNet-64, DiT-S target / DiT-Nano draft, 256 tokens)

**Per-timestep AURC (reliability).** ε-agreement equals the oracle and cuts AURC by
58% vs random; frequency-token and token-norm stay near random.

**FID-10k budget sweep** (val reference; lower is better):

| target-token budget | 70% | 50% | 30% | 10% |
|---|---|---|---|---|
| Target-only | 42.68 | 43.22 | 42.49 | 42.89 |
| **FreqSpec-token** | **46.81** | **52.00** | **62.06** | **78.69** |
| Random-token | 57.40 | 68.07 | 79.84 | 89.23 |
| Draft-only | 92.41 | 92.44 | 93.12 | 92.84 |

FreqSpec-token < Random-token at every tested budget (ordering
Target < FreqSpec < Random < Draft throughout).

**Selector ablation (accept 0.7).** ε-L2 61.7, ε-cosine 62.2 ≪ frequency 79.9,
token-norm 79.4, random 79.6 — prediction agreement is the binding signal.

**Seed repeat (accept 0.7, 3 seeds).** FreqSpec 61.70 ± 0.15 vs Random 79.30 ± 0.17
(gap two orders of magnitude larger than the seed variance).

---

## Notes

- The DiT study is a validation of the verifier, **not** a state-of-the-art
  generator; DiT-S on ImageNet-64 is intentionally modest.
- Exact hyperparameters and the full experiment log are described in the paper and
  its appendix.