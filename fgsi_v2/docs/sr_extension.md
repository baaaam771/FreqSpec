# FreqSpec-SR: Super-Resolution Instantiation (AAAI track)

This note describes the super-resolution (SR) extension of FreqSpec and how it
fits the AAAI generalization plan. The goal is to demonstrate that FreqSpec is
not an inpainting-specific trick but a general **region-wise speculative
verification** framework for conditional diffusion inference. SR is the first
additional task because it shares inpainting's core property — spatially
non-uniform difficulty — while removing the mask, which forces the framework to
stand on its frequency-guided verifier rather than on mask geometry.

## 1. The generalization (method level)

The WACV paper frames acceleration as: *which masked-region patches need the
full target, and which can be locally approximated by a compact draft under
verification?* The AAAI framing drops "masked-region" and asks the question over
a general **verification region** Ω. Inpainting and SR are two instantiations of
the identical verifier:

| Component            | Inpainting (WACV)              | Super-resolution (AAAI)          |
|----------------------|--------------------------------|----------------------------------|
| Condition `c`        | mask + masked-image latent     | low-resolution image             |
| Verification region Ω| inside the hole (mask = 1)     | the whole latent field (ones)    |
| Target               | SDXL-Inpainting (2.6B, 9ch UNet)| SD x4-upscaler (frozen, 7ch UNet)|
| Draft                | 82M U-Net, 9ch in              | 82M U-Net, 7ch in (z 4 + LR 3)   |
| Saliency             | wavelet + boundary + interior  | wavelet only (HF = where target needed) |
| Known-region blend   | on (re-inject known pixels)    | off (LR anchor enters via cond)  |

Everything else is unchanged: the ε-agreement test, predicted-x̂₀ disagreement
gate, soft patch blending, timestep-dependent strictness, and drift-aware
K-step gating are all defined per patch over Ω and are therefore task-agnostic.
This is the central AAAI claim: **the verifier is the contribution, and it is
backbone- and task-portable.** SR makes the point cleanly because, with Ω the
whole field and no mask, accept-on-low-frequency / verify-on-high-frequency is
exactly the frequency-guided story the WACV title promises.

## 2. What is implemented

The WACV-validated inpainting path (`inference/speculative.py`, `models/
target_wrapper.py`) is untouched. New, additive files:

- `inference/speculative_general.py` — `fgsr_refine` / `baseline_refine`: the
  generalized verifier over a `region_z`. Reproduces the inpainting sampler
  when `region_z = mask`, `blend_known = True`; runs SR when `region_z = ones`,
  `blend_known = False`, wavelet-only saliency. Keeps the full instrumentation
  (`return_usage_map`, `collect_patch_logs`) so the AAAI verifier-reliability
  table and qualitative usage maps come for free.
- `models/sr_target_wrapper.py` — `SRTargetWrapper` for the x4 upscaler with the
  unified `predict_eps(z_t, t, cond_lr, region, …, noise_level)` interface and a
  dummy fallback. Handles the upscaler's noise-level class conditioning.
- `models/draft.py` — generalized: `DraftEpsUNet(cond_ch, use_mask)`. Inpainting
  default (`cond_ch=4, use_mask=True`) is byte-for-byte unchanged (9ch, 82.35M);
  SR uses `cond_ch=3, use_mask=False` (7ch, 82.35M).
- `inference/run_sr.py` — end-to-end SR runner (baseline vs FreqSpec-SR + PSNR /
  SSIM / HH-band-PSNR vs HR ground truth).
- `training/train_sr.py` — SR draft training; region-aware loss reduces to
  target-distillation on low-frequency and GT-supervision on high-frequency.
- `run_sr_freqspec.sh`, `smoke_sr.py` — server run script and dummy smoke tests
  (all passing here without GPU/weights).

The whole pipeline runs end-to-end in dummy mode, and a controlled check
confirms the lookahead mechanism still fires on the SR path: when the draft
agrees with the target, target NFE drops below the step count (1.67× target
speedup in the synthetic check), exactly as in inpainting.

## 3. SR-specific design decisions

The x4 upscaler conditions on the **low-res image itself** (3 RGB channels,
concatenated with the 4-channel noisy latent at latent resolution), not a VAE
latent — hence `cond_ch=3`. The LR image is noised once to `low_res_noise_level`
(default 20) and that level is passed to the UNet as a class label; the wrapper's
`prepare_lr_cond` does this and returns the noised cond plus the level tensor,
forwarded opaquely through the sampler via `target_extra`. Because the LR anchor
enters through the model input, there is no known-region re-injection at sampling
time, so `blend_known=False`. Saliency is the pure LWD wavelet HF map; with Ω the
whole field, the boundary and interior terms are meaningless (a full-field region
has no boundary), so they are switched off — which, conveniently, is the most
honest standalone test of the frequency prior.

## 4. Experiment plan

**Datasets (standard SR benchmarks).** DIV2K validation as the main set; Set5 /
Set14 / BSD100 / Urban100 as small standard test sets. Bicubic ×4 degradation to
match the upscaler's training. Start with DIV2K (n≈100) for the operating-point
table, then add the small classics for the cross-set table.

**Baselines.** The AAAI reviewer's reflex is "just use fewer steps." The table
must pre-empt that, mirroring the WACV Table 2 structure:

| Method                    | Role                                   |
|---------------------------|----------------------------------------|
| Target 50-step            | quality upper bound (reference)        |
| Target {25, 30, 40}-step  | global step-reduction reference curve  |
| DPM-Solver / DDIM short   | solver-based acceleration              |
| Draft-only                | cost lower bound (no verification)     |
| FreqSpec-SR strict/mid/default | verifier-controlled operating points |

Report, per method: target NFE, accepted patch rate, wall-clock speedup vs the
50-step target, PSNR, SSIM, LPIPS, and HH-band PSNR (high-frequency fidelity —
the metric SR most cares about and where the verifier story is sharpest), plus
LPIPSₜ divergence from the 50-step target.

**Headline analyses to port from WACV.** (1) Verifier reliability over millions
of patches via `collect_patch_logs` → risk-coverage / AURC, with the same
selector comparison (random / saliency-only / ε-only / full / x̂₀-oracle); SR
should show the full verifier near the oracle just as inpainting did. (2) The
single-knob tolerance trade-off (strict/mid/default). (3) Input-adaptive target
NFE: smoother / low-frequency images should accept more and spend fewer target
calls — and here the correlate is image high-frequency content, which is the
natural SR difficulty axis, likely a *cleaner* correlation than the COCO result.

**Positioning.** Identical to WACV: FreqSpec-SR is not pitched as beating
reduced-step sampling on global PSNR at matched speed, but as a controllable,
region-selective verifier that exposes where the expensive upscaler is actually
needed (high-frequency detail) and falls back safely elsewhere.

## 5. Server verification checklist

This environment has no GPU or HF access, so the x4-upscaler interface is modeled
from the diffusers `StableDiffusionUpscalePipeline` and must be confirmed once on
the server before large runs. The wrapper already prints warnings if any check
fails:

1. `unet.config.in_channels == 7` (4 latent + 3 LR). If different, adjust the
   `cond` assembly in `predict_eps`.
2. `scheduler.config.prediction_type == "epsilon"`. The generalized DDIM step
   assumes ε-prediction (same as inpainting). If it is v-prediction, add a v→ε
   conversion in `scheduler.ddim_step` or the wrapper.
3. `vae.config.scaling_factor` (expected ≈ 0.08333) and that `vae.decode`
   upsamples ×4 (latent 128 → image 512). The runner assumes latent spatial size
   == `lr_size` and HR == `lr_size × scale`.
4. The upscaler's text encoder is a single CLIP encoder (SD2-style); the wrapper
   uses the single-encoder path. SR runs typically use `guidance_scale` ≈ 1, so
   text conditioning is minor — confirm whether a generic prompt helps.

Once 1–3 pass, `run_sr_freqspec.sh <hr_image> <draft_ckpt>` should produce a
meaningful baseline-vs-FreqSpec comparison on one image.

## 6. Training recipe

Train one draft on the SR distribution (DIV2K train, optionally + Flickr2K),
HR = `lr_size × scale` (e.g. 512 for lr_size 128), bicubic ×4 LR conditioning,
noise_level 20, ε-space U-Net at 82.35M to match the inpainting draft. The loss
distills toward the target on low-frequency regions and supervises with
ground-truth noise on high-frequency regions (wavelet time-mask, `ℓ = 0.3`),
weights `(α_distill, γ_main, λ_uniform) = (0.5, 2.0, 1.0)` as in the paper. AdamW
1e-4, grad-clip 1.0, EMA 0.999, ~400k steps for parity with the FFHQ/COCO drafts.

```
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
python training/train_sr.py \
  --data_root /mnt/HDD_12TB/bam_ki/datasets/div2k/train \
  --out_dir   /mnt/HDD_12TB/bam_ki/ckpt_sr \
  --lr_size 128 --scale 4 --noise_level 20 \
  --batch_size 4 --max_steps 400000 --use_ema
```

## 7. Running the experiments

After training an SR draft, the full pipeline is one script:

```
export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
./run_sr_experiments.sh <div2k_valid> <ckpt_sr/draft_sr_final.pt> <out_root> 100
```

which runs, and aggregates:

- `sr_baseline_sweep.py` -> `analyze_sr.py`: the operating-point table
  (`sr_table.{csv,tex}`) — target_s{50,40,30} vs FreqSpec strict/mid/default,
  paired per image, with NFE / accept / wall-speedup / PSNR / SSIM / HH-PSNR /
  LPIPS / LPIPSt (LPIPSt is divergence from the per-image target_s50 output).
- `sr_verifier_reliability_sweep.py` -> `analyze_verifier_reliability.py`: the
  verifier-reliability table (AURC / risk-coverage / FAR). The SR sweep dumps
  patch logs in the exact format the inpainting analyzer expects, so the **same
  analyzer runs unchanged** — including the wavelet-only selector, which is the
  SR-relevant frequency prior. Because SR verifies the whole field (no mask),
  every patch is logged, so patch counts per image are much larger than
  inpainting.

Both sweeps call `fgsr_refine` (which routes through `freqspec_core`), so the
acceptance logic is identical to inpainting; only the task setup differs.

## 8. Beyond SR (supplement)

Per the AAAI plan, a DiT token-wise proof-of-concept is the natural supplement:
the same acceptance rule applies per latent patch-token instead of per spatial
cell. `fgsr_refine` already treats the region abstractly, so a DiT instantiation
mainly needs a token-grid analogue of the patch pooling and a DiT target/draft
wrapper pair. That is out of scope for this first deliverable but the generalized
verifier was written to accommodate it.
