"""
Frequency-Guided Speculative Refinement (FGSR) for ε-prediction inpainting.

Target은 frozen pretrained SD-Inpainting, Draft는 학습된 작은 모델.
두 모델 모두 동일한 ε-prediction 인터페이스를 가지므로 직접 비교 가능.

Algorithm (한 DDIM step at timestep t):
1. t > t_spec_start_norm: target으로만 진행 (saliency 안정화)
2. t ≤ t_spec_start_norm: speculative phase
   a. 현재 z_t에서 saliency 계산
   b. ε_target = target(z_t, t, cond, mask)
   c. ε_draft  = draft (z_t, t, cond, mask)
   d. patch별 agreement score: a = exp(-β ||ε_target - ε_draft||²)
   e. patch별 tolerance: tol = tol_low + (tol_high - tol_low)*(1-saliency)
   f. accept patch가 어디인지 결정
   g. z_t_next 합성:
       - accept (평탄): draft ε로 DDIM step
       - reject (어려움): target ε로 DDIM step
3. Accept rate가 높으면 draft로 K step 추가 진행 (target call 절약)

Option B (default): velocity-agreement = "distribution-aware refinement"
Option A (stub): divergence-based density ratio (Hutchinson estimator)
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.wavelet import DWT2D, combined_saliency
from training.scheduler import DDPMSchedule


# ============================================================
# Helpers
# ============================================================
def patch_agreement(eps_a, eps_b, patch_size: int, beta: float):
    """Per-patch agreement: a = exp(-β · mean_channel ||eps_a - eps_b||²)."""
    diff_sq = (eps_a - eps_b).pow(2).mean(dim=1, keepdim=True)         # [B,1,H,W]
    diff_patch = F.avg_pool2d(diff_sq, patch_size, stride=patch_size)  # [B,1,Hp,Wp]
    return torch.exp(-beta * diff_patch)


# ============================================================
# Main FGSR inference
# ============================================================
@torch.no_grad()
def fgsr_inpaint(
    target,                        # TargetWrapper instance
    draft,                         # DraftEpsUNet
    z_init: torch.Tensor,          # [B,4,Hl,Wl] noise latent
    cond_z: torch.Tensor,          # [B,4,Hl,Wl] masked-image latent
    mask_z: torch.Tensor,          # [B,1,Hl,Wl]
    scheduler: DDPMSchedule,
    num_inference_steps: int = 50,
    # FGSR hyperparams
    K: int = 3,
    patch_size: int = 4,
    t_spec_start_norm: float = 0.7,
    beta: float = 10.0,
    tol_low: float = 0.05,
    tol_high: float = 0.5,
    boundary_weight: float = 1.0,
    dwt=None,
    verbose: bool = False,
):
    """
    Returns:
        z_final, stats
    """
    device = z_init.device
    if dwt is None:
        dwt = DWT2D("haar").to(device)

    ts = scheduler.get_ddim_schedule(num_inference_steps)  # [N] descending int
    N = len(ts)
    z = z_init.clone()
    B, C, H, W = z.shape

    target_calls = 0
    draft_calls = 0
    total_proposed = 0
    total_accepted = 0
    spec_steps = 0
    stab_steps = 0

    i = 0
    while i < N:
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if (i + 1) < N else -1

        # normalized timestep for saliency threshold
        t_norm = t_cur / (scheduler.num_train_timesteps - 1)

        t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

        # ---------------- Phase 1: stabilization ----------------
        if t_norm > t_spec_start_norm:
            eps_t = target.predict_eps(z, t_tensor, cond_z, mask_z)
            target_calls += 1
            z, _ = scheduler.ddim_step(z, eps_t, t_cur, t_prev)
            stab_steps += 1
            if verbose:
                print(f"[stab] i={i} t={t_cur} (target only)")
            i += 1
            continue

        # ---------------- Phase 2: speculative refinement ----------------
        spec_steps += 1

        # (a) current saliency from z_t (iterative)
        sal = combined_saliency(z, mask_z, dwt,
                                boundary_weight=boundary_weight,
                                target_size=(H, W))
        sal_patch = F.avg_pool2d(sal, patch_size, stride=patch_size)
        tol_patch = tol_low + (tol_high - tol_low) * (1 - sal_patch)

        # (b) target + draft eps at current step
        eps_t = target.predict_eps(z, t_tensor, cond_z, mask_z)
        target_calls += 1
        eps_d = draft(z, t_tensor, cond_z, mask_z)
        draft_calls += 1

        # (c) per-patch agreement and accept decision
        a_patch = patch_agreement(eps_d, eps_t, patch_size, beta)
        accept_patch = (a_patch > (1 - tol_patch)).float()  # [B,1,Hp,Wp]

        # 마스크 외부 patch는 어차피 보존되므로 강제 accept (계산 절약)
        mask_patch = F.max_pool2d(mask_z, patch_size, stride=patch_size)
        accept_patch = torch.where(mask_patch < 0.5,
                                   torch.ones_like(accept_patch),
                                   accept_patch)

        accept_full = F.interpolate(accept_patch, size=(H, W), mode="nearest")
        n_patches = accept_patch.numel()
        n_acc = accept_patch.sum().item()
        total_proposed += n_patches
        total_accepted += n_acc
        accept_rate = n_acc / max(n_patches, 1)

        # (d) DDIM step with selective eps
        # accept = draft eps 사용, reject = target eps 사용
        eps_mix = accept_full * eps_d + (1 - accept_full) * eps_t
        z_next, _ = scheduler.ddim_step(z, eps_mix, t_cur, t_prev)

        # (e) accept rate가 높으면 draft로 K step 추가 진행 (target 호출 절약)
        if accept_rate > 0.6 and K > 1 and (i + K) <= N:
            # 추가 K-1 step을 draft 단독으로 진행 (이미 1 step은 위에서 진행됨)
            z = z_next
            for k in range(1, K):
                if (i + k + 1) > N: break
                t_cur_k = int(ts[i + k].item())
                t_prev_k = int(ts[i + k + 1].item()) if (i + k + 1) < N else -1
                t_tensor_k = torch.full((B,), t_cur_k, device=device, dtype=torch.long)
                eps_d_k = draft(z, t_tensor_k, cond_z, mask_z)
                draft_calls += 1
                # 마스크 외부는 그대로 두기 위해 hole 영역에만 적용
                z_step, _ = scheduler.ddim_step(z, eps_d_k, t_cur_k, t_prev_k)
                z = z_step
            advanced_k = K
            if verbose:
                print(f"[spec-K] i={i}..{i+K-1} t={t_cur} "
                      f"accept={accept_rate:.2f}")
            i += advanced_k
        else:
            z = z_next
            if verbose:
                print(f"[spec-1] i={i} t={t_cur} accept={accept_rate:.2f}")
            i += 1

    stats = {
        "target_calls": target_calls,
        "draft_calls": draft_calls,
        "nfe_total": target_calls + draft_calls,
        "nfe_target_baseline": N,
        "accept_rate": total_accepted / max(total_proposed, 1),
        "stab_steps": stab_steps,
        "spec_steps": spec_steps,
        "target_speedup": N / max(target_calls, 1),
    }
    return z, stats


# ============================================================
# Baseline: target-only DDIM
# ============================================================
@torch.no_grad()
def baseline_inpaint(target, z_init, cond_z, mask_z, scheduler, num_inference_steps=50):
    ts = scheduler.get_ddim_schedule(num_inference_steps)
    N = len(ts)
    z = z_init.clone()
    B = z.shape[0]
    for i in range(N):
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if (i + 1) < N else -1
        t_tensor = torch.full((B,), t_cur, device=z.device, dtype=torch.long)
        eps_t = target.predict_eps(z, t_tensor, cond_z, mask_z)
        z, _ = scheduler.ddim_step(z, eps_t, t_cur, t_prev)
    return z, {"target_calls": N, "nfe_total": N, "accept_rate": None}


if __name__ == "__main__":
    from models.target_wrapper import TargetWrapper
    from models.draft import DraftEpsUNet
    torch.manual_seed(0)
    device = "cpu"
    target = TargetWrapper("dummy/none", device=device)
    draft = DraftEpsUNet(latent_ch=4).to(device).eval()
    sch = DDPMSchedule(device=device)

    z_init = torch.randn(1, 4, 32, 32, device=device)
    cond = torch.zeros(1, 4, 32, 32, device=device)
    mask = (torch.rand(1, 1, 32, 32, device=device) > 0.7).float()

    z_b, s_b = baseline_inpaint(target, z_init, cond, mask, sch, num_inference_steps=20)
    z_s, s_s = fgsr_inpaint(target, draft, z_init, cond, mask, sch,
                            num_inference_steps=20, K=3, patch_size=4,
                            verbose=True)
    print("baseline stats:", s_b)
    print("fgsr     stats:", s_s)
