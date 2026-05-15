# FGSI v2: Frozen-Target Frequency-Guided Speculative Inpainting

기존 v1과의 차이: **Target은 사전학습 SOTA (SD-Inpainting)를 frozen으로 사용**,
**Draft만 LWD-style frequency-guided supervision으로 학습**.

이 구조의 장점:
- 학습 비용 ↓ (target 학습 안 함, draft만 학습)
- Target 품질이 SOTA로 보장됨
- **"any latent diffusion inpainting model에 plug-and-play로 efficient inference를 더한다"** 라는
  깔끔한 contribution statement

## Contribution Hierarchy

| Component | Source | Status |
|---|---|---|
| Stable Diffusion Inpainting | StabilityAI | Borrowed as target |
| Scale-consistency VAE | LWD (선택) | Optional VAE swap |
| Wavelet energy saliency (LWD Eq.3) | LWD | Borrowed |
| Time-dependent supervision mask (LWD Eq.6) | LWD | Borrowed for draft training |
| **Mask-boundary-aware saliency** | **Ours** | Inpainting extension |
| **Frequency-guided draft training** | **Ours** | LWD를 distillation에 적용 |
| **ε-space speculative refinement** | **Ours** | Main novelty |
| Patch-wise verify with adaptive tolerance | **Ours** | |

## Directory
```
fgsi_v2/
├── models/
│   ├── wavelet.py         # DWT, LWD saliency, boundary-aware
│   ├── target_wrapper.py  # SD-Inpainting frozen wrapper
│   └── draft.py           # DraftEpsUNet (small, ε-prediction)
├── training/
│   ├── scheduler.py       # DDPM/DDIM (SD-호환)
│   ├── losses.py          # DraftLoss (distill + main + uniform)
│   └── train.py           # draft-only training
├── inference/
│   ├── speculative.py     # fgsr_inpaint (ε-space)
│   └── run_inpaint.py     # baseline vs FGSR
└── utils/metrics.py
```

## 핵심 동작 원리

### Draft Training (target frozen)

매 step:
```
GT image → VAE → z0
mask, eps_gt, t 샘플링
z_t = √α̅_t · z0 + √(1-α̅_t) · eps_gt

eps_target = target.predict_eps(z_t, t, cond, mask)   [frozen, no grad]
eps_draft  = draft (z_t, t, cond, mask)                [trainable]

A_combined = LWD_saliency(z0) + λ_b · BoundaryIndicator(mask)
M_t = 1[A_combined + ℓ ≥ t/T]

L = α_distill · (1 - M_t) · ||eps_draft - eps_target||²    ← 평탄 영역: target 모방
  + γ_main    · M_t       · ||eps_draft - eps_gt    ||²    ← 어려운 영역: GT 학습
  + λ_uniform              · ||eps_draft - eps_gt    ||²    ← 안전망

backward & update draft only
```

이렇게 학습하면:
- **평탄 영역**에서 draft는 target과 거의 동일한 ε를 출력 → 추론 시 accept↑
- **어려운 영역**에서 draft는 GT를 직접 학습 (target에 너무 의존하지 않음)

### Speculative Refinement (Inference)

DDIM sampling을 진행하면서:
```
for t in DDIM_schedule:
    if t > t_spec_start:
        # 초기 noisy phase: target만 사용 (saliency 신뢰 어려움)
        eps = target(z_t, t, cond, mask)
        z = DDIM_step(z, eps)
    else:
        # speculative phase
        sal = combined_saliency(z_t, mask)  # iterative saliency
        eps_t = target(z_t, ...)
        eps_d = draft (z_t, ...)
        
        a = exp(-β · ||eps_t - eps_d||²) per patch
        tol = tol_low + (tol_high - tol_low) * (1 - sal_patch)
        accept = (a > 1 - tol)
        
        eps_mix = accept · eps_d + (1-accept) · eps_t
        z = DDIM_step(z, eps_mix)
        
        if accept_rate > 0.6:
            # K step 더 draft 단독 진행 (target call 절약)
            for k in 1..K-1: z = DDIM_step(z, draft(z, ...))
```

## Setup
```bash
pip install torch torchvision diffusers transformers accelerate Pillow numpy
# pywavelets 선택 (없어도 Haar/db2 fallback)
```
처음 사용 시 SD-Inpainting checkpoint 자동 다운로드 (~5GB).

## Training (draft only)
```bash
python -m training.train \
    --data_root /path/to/Places2 \
    --target_id stabilityai/stable-diffusion-2-inpainting \
    --image_size 512 --batch_size 2 \
    --alpha_distill 1.0 --gamma_main 1.0 --lambda_uniform 0.1 \
    --boundary_weight 1.0 --ell 0.3 \
    --out_dir runs/draft_v1
```
- Target은 GPU에 frozen으로 상주 (~3GB VRAM)
- Draft만 학습 (~500MB VRAM)
- A100 1장에서 batch_size=4까지 가능

## Inference
```bash
python -m inference.run_inpaint \
    --image sample.jpg --mask sample_mask.png \
    --draft_ckpt runs/draft_v1/draft_e9.pt \
    --num_steps 50 --K 3 --patch 4 \
    --t_spec_start 0.7 --beta 10.0 \
    --tol_low 0.05 --tol_high 0.5 \
    --verbose --out_dir results/
```

## 실험 권장

| Setup | 의미 | 기대 결과 |
|---|---|---|
| SD-Inpainting 50 DDIM | Baseline | FID = X, NFE_target = 50 |
| SD-Inpainting 25 DDIM | Naive fewer-step | FID = X + Δ, NFE_target = 25 |
| **FGSI (ours)** | Frequency-guided spec | FID ≈ X, NFE_target ≈ 25-30 |
| FGSI saliency-blind | Speculative w/o frequency | FID = X + ε, accept rate↓ |

**핵심 주장 가능 명제**:
- "Same FID at fewer target NFE" → quality-preserving speedup
- "Higher quality at same target NFE" → quality boost at no extra cost

## Hyperparameter 튜닝 가이드

| param | 효과 | 시작값 |
|---|---|---|
| `alpha_distill` | 평탄 영역에서 target 모방 강도 | 1.0 |
| `gamma_main` | 어려운 영역에서 GT 학습 강도 | 1.0 |
| `lambda_uniform` | 전 영역 안전망 | 0.1 |
| `ell` | LWD lower bound (Eq.6) | 0.3 |
| `boundary_weight` | 경계 saliency 강조 | 1.0 |
| `t_spec_start` | spec 시작 timestep (normalized) | 0.7 |
| `K` | accept 시 draft 추가 step 수 | 3 |
| `beta` | agreement score 민감도 | 10 |
| `tol_low / tol_high` | strict / lenient tolerance | 0.05 / 0.5 |

## 알려진 제약

- Target은 latent diffusion 계열로 한정 (LaMa, MAT 같은 픽셀 공간 모델 불가)
- Draft를 target과 같은 parameterization(ε-prediction)으로 학습해야 함
- Pretrained target의 성능이 천장 — target이 약하면 ours도 약함
- 초기 timestep (t > t_spec_start)에서는 speculative 미적용 → 그 구간은 baseline과 동일

## Citation

LWD를 학습 supervision에 차용:
```bibtex
@article{sigillo2025lwd,
  title={Latent Wavelet Diffusion: Enabling 4K Image Synthesis for Free},
  author={Sigillo, Luigi and He, Shengfeng and Comminiello, Danilo},
  journal={arXiv preprint arXiv:2506.00433},
  year={2025}
}
```
Target으로 SD-Inpainting 사용:
```bibtex
@article{rombach2022ldm,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and others},
  journal={CVPR},
  year={2022}
}
```
