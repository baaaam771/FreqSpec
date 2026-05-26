# FreqSpec-Inpaint

**Frequency-Guided Speculative Refinement for Latent Diffusion Inpainting**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](docs/paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)

Official PyTorch implementation of **FreqSpec-Inpaint**, a frequency-guided speculative refinement framework that delivers **1.28–1.62× wall-clock speedup** on Stable Diffusion 2 / SDXL Inpainting while preserving baseline-level perceptual quality. A single tolerance hyperparameter controls the quality-speed trade-off at inference time *without retraining*.

<!-- TODO: Add teaser figure here -->
<!-- ![Teaser](docs/teaser.png) -->

## ✨ Key Features

- **Controllable Pareto trade-off**: One tolerance hyperparameter spans LPIPS 0.007–0.115 across 1.02–1.64× speedup.
- **Plug-and-play with frozen target**: Works with pretrained SD2-Inpainting and SDXL-Inpainting without modifying the target model.
- **Lightweight draft**: Only **82M parameters** (~30× smaller than SDXL).
- **Inpainting-specific saliency**: Wavelet-based saliency extended with a mask-boundary indicator for seam-aware acceptance.
- **Cross-domain validated**: Tested on Places2, FFHQ, and COCO 2017 with a 3×3 transferability matrix.

## 📊 Main Results

### Pareto Frontier (SDXL Inpainting, 1024×1024)

| Dataset | Tolerance | Speedup | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| Places2 | default | 1.35× | 22.08 | 0.948 | 0.074 |
| Places2 | strict | 1.05× | 25.88 | 0.977 | **0.029** |
| FFHQ | default | 1.62× | 22.44 | 0.934 | 0.085 |
| FFHQ | strict | 1.12× | 27.58 | 0.972 | **0.025** |
| COCO (captions) | default | 1.23× | 21.06 | 0.936 | 0.077 |

See [paper](docs/paper.pdf) Table 1 for the full table including SD2 results.

## 🔧 Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher with CUDA 11.8+
- ~16 GB GPU memory for SD2 (512×512), ~36 GB for SDXL (1024×1024)
- Tested on NVIDIA RTX Pro 6000 (48GB)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<YOUR-USERNAME>/freqspec-inpaint.git
cd freqspec-inpaint

# 2. Create a conda environment
conda create -n FreqSpec python=3.10
conda activate FreqSpec

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt
```

### Download Pretrained Target Models

```bash
# SD2 Inpainting (865M, ~5 GB)
hf download stabilityai/stable-diffusion-2-inpainting \
    --local-dir checkpoints/stable-diffusion-2-inpainting

# SDXL Inpainting (2.6B, ~12 GB)
hf download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --local-dir checkpoints/stable-diffusion-xl-1.0-inpainting-0.1
```

> **Note**: `huggingface-cli` is deprecated as of `huggingface_hub` v1.0 (Oct 2025). Use `hf <resource> <action>` format.

### Download Pretrained Draft Models

<!-- TODO: Upload to Hugging Face and update these links -->

| Draft | Target | Dataset | Steps | Link |
|---|---|---|---|---|
| Places2 draft | SD2-Inpainting | Places2 | 150K | [TBA](#) |
| Places2 draft | SDXL-Inpainting | Places2 | 290K | [TBA](#) |
| FFHQ draft | SDXL-Inpainting | FFHQ | 200K | [TBA](#) |
| COCO draft | SDXL-Inpainting | COCO 2017 | 200K | [TBA](#) |

```bash
# Example: download SDXL Places2 draft
mkdir -p checkpoints/drafts
# wget <CHECKPOINT_URL> -O checkpoints/drafts/sdxl_places2_draft.pt
```

## 🚀 Quick Start: Inference

Inpaint a single image with default settings:

```bash
python -m inference.run_inpaint \
    --target_id checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \
    --draft_ckpt checkpoints/drafts/sdxl_places2_draft.pt \
    --image path/to/image.jpg \
    --out_dir results/demo \
    --prompt "a photograph" \
    --K 3 --tol_low 0.03 --tol_high 0.3 \
    --use_ema_draft
```

### Key Inference Arguments

| Argument | Default | Description |
|---|---|---|
| `--K` | `3` | Speculation window (1=single-step, 3=balanced, 5=aggressive) |
| `--tol_low` | `0.03` | Loose tolerance for low-saliency (smooth) regions |
| `--tol_high` | `0.3` | Strict tolerance for high-saliency (texture, boundary) regions |
| `--t_spec_start` | `0.7` | Phase 1 (target-only) duration. Larger = more conservative |
| `--patch` | `4` | Patch size for per-patch acceptance |
| `--beta` | `10.0` | Acceptance sharpness (lower = looser) |
| `--guidance_scale` | `7.5` | CFG scale (SDXL); use `1.0` for SD2 |
| `--num_steps` | `50` | Denoising steps |

### Tolerance Presets

For convenience, use these presets:

```bash
# Strict (≈1.05×, near-baseline quality)
--tol_low 0.01 --tol_high 0.1

# Mid (≈1.11×, balanced)
--tol_low 0.02 --tol_high 0.15

# Default (≈1.35×, recommended)
--tol_low 0.03 --tol_high 0.3

# Aggressive (≈1.64×, for previews)
--tol_low 0.03 --tol_high 0.3 --t_spec_start 0.9
```

### Per-Image Captions (COCO-style)

For COCO-trained drafts, use per-image captions for best quality (20% LPIPS improvement over generic prompts):

```bash
python -m inference.run_inpaint \
    --target_id checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \
    --draft_ckpt checkpoints/drafts/sdxl_coco_draft.pt \
    --image path/to/coco_image.jpg \
    --prompt "A young boy holding a baseball bat at home plate" \
    --out_dir results/demo
```

## 🎯 Training Your Own Draft

### Prepare Datasets

```bash
# Places2 (1.8M images, ~120 GB)
# Download from: http://places2.csail.mit.edu/

# FFHQ (70K images, ~90 GB at 1024×1024)
hf download <USERNAME>/ffhq_hf --repo-type dataset \
    --local-dir datasets/ffhq_hf

# COCO 2017
# Download from: https://cocodataset.org/#download
# Required: train2017.zip, val2017.zip, annotations_trainval2017.zip
```

### Train Draft

```bash
# Train SDXL Places2 draft (~6 days on RTX Pro 6000)
python -m train_freqspec \
    --target_id checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \
    --dataset_root datasets/places2 \
    --image_size 1024 \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1e-4 \
    --total_steps 290000 \
    --use_cfg --guidance_scale 7.5 \
    --out_dir runs/sdxl_places2 \
    --save_every 10000
```

### Train with COCO Captions

```bash
python -m train_freqspec_coco \
    --target_id checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \
    --dataset_root datasets/coco2017 \
    --captions_json datasets/coco2017/annotations/captions_train2017.json \
    --image_size 1024 \
    --total_steps 200000 \
    --out_dir runs/sdxl_coco
```

### Training Loss

```
L = α_d · L_distill + γ_m · L_main + λ_u · L_uniform
```

Where:
- `L_distill`: Saliency-weighted distillation against target's ε prediction
- `L_main`: Standard noise prediction loss
- `L_uniform`: Regularization on full image
- Default weights: `α_d = 1.0`, `γ_m = 1.0`, `λ_u = 0.1`

## 📁 Repository Structure

```
freqspec-inpaint/
├── freqspec/              # Core library
│   ├── saliency.py        # Wavelet + boundary saliency
│   ├── speculative.py     # ε-space speculative refinement
│   ├── acceptance.py      # Per-patch acceptance logic
│   └── draft_unet.py      # 82M draft U-Net architecture
├── inference/
│   ├── run_inpaint.py     # Main inference script
│   └── eval_dataset.py    # Batch evaluation
├── training/
│   ├── train_freqspec.py        # Training loop (no captions)
│   └── train_freqspec_coco.py   # Training with captions
├── evaluation/
│   ├── metrics.py         # PSNR, SSIM, LPIPS
│   └── summarize.py       # Aggregate results, make figures
├── figures/               # Paper figures (regenerated from results)
├── docs/
│   ├── paper.pdf          # Paper
│   └── teaser.png
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔬 Reproducing Paper Results

### Main Pareto results (Table 1, Figure 2)

```bash
bash scripts/eval_main_pareto.sh
```

This runs default / mid / strict tolerance on each model-dataset combination (~4 hours on RTX Pro 6000).

### Cross-domain transferability (Table 2, Figure 3)

```bash
bash scripts/eval_cross_domain.sh
```

### Ablations (Figure 4)

```bash
bash scripts/eval_ablations.sh
```

Outputs are aggregated into `results/final_all_results.csv` and plotted by `evaluation/summarize.py`.

## 📈 Practical Deployment Guide

Based on our 32-measurement empirical analysis, we recommend:

| Use Case | Tolerance | Speedup | Notes |
|---|---|---|---|
| **Portrait editing** | strict | 1.12× | Use FFHQ-trained draft (LPIPS 0.025) |
| **General editor** | default | 1.35× | Use COCO-trained draft with per-image captions |
| **Human-reviewed output** | strict | 1.05× | LPIPS 0.029, near-baseline |
| **Batch processing** | default | 1.35× | Best speedup/quality balance |
| **Previews & iteration** | aggressive | 1.64× | Re-run with strict once satisfied |

**Cross-domain tips**:
- Natural scenes (Places2) is a "universal target" — any sufficiently trained draft works
- Faces (FFHQ) require *domain-specific* training; cross-domain drafts fail badly (LPIPS 0.153)
- For unknown-domain deployment, train on COCO (most diverse) with per-image captions

## ⚠️ Limitations

- **Comparison with distillation methods** (LCM, SDXL-Turbo): Not included; FreqSpec is *complementary* to distillation and could be combined.
- **Newer architectures**: Tested on U-Net based SD2/SDXL. Diffusion Transformer (DiT) models (SD3, FLUX) may need redesign.
- **Single-target compatibility**: Draft is trained per target; multi-target distillation is future work.

See paper Section 5.4 for full discussion.

## 📜 Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{lee2025freqspec,
  title   = {FreqSpec-Inpaint: Frequency-Guided Speculative Refinement
             for Latent Diffusion Inpainting},
  author  = {Lee, BeomGi},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```

## 🙏 Acknowledgments

This work builds on several open-source projects:

- [Diffusers](https://github.com/huggingface/diffusers) — Stable Diffusion infrastructure
- [Latent Wavelet Diffusion (LWD)](https://arxiv.org/abs/2506.00433) — Wavelet saliency formulation (we extend with boundary-awareness)
- [Stable Diffusion 2 Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) — Target model
- [SDXL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) — Target model

## 📧 Contact

For questions, issues, or collaboration:

- **Author**: BeomGi Lee  
- **Email**: jeongiun@hanyang.ac.kr  
- **GitHub Issues**: For bug reports and feature requests, please open an [issue](https://github.com/<YOUR-USERNAME>/freqspec-inpaint/issues)

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<sub>Last updated: 2025-XX-XX</sub>
