"""
Frequency-Guided Speculative Refinement (FGSR) for ε-prediction inpainting.

[PATCHED]: adds optional `return_usage_map=True` which collects the
per-verified-timestep soft-blend weight w(p) (or hard accept indicator)
and returns the time-averaged usage map in stats["usage_map"].
The map has shape [B, 1, H_lat, W_lat], values in [0, 1], where 1 means
"draft fully used" and 0 means "target fallback". The mask-outside
region is excluded from the average (always 0).

Target은 frozen pretrained SD-Inpainting, Draft는 학습된 작은 모델.
두 모델 모두 동일한 ε-prediction 인터페이스를 가지므로 직접 비교 가능.
"""
import os
import sys
import torch
import torch.nn.functional as F
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.wavelet import DWT2D, combined_saliency
from training.scheduler import DDPMSchedule


# ============================================================
# Helpers
# ============================================================
def patch_agreement(eps_a, eps_b, patch_size: int, beta: float):
    """Per-patch agreement: a = exp(-β · mean_channel ||eps_a - eps_b||²)."""
    diff_sq = (eps_a - eps_b).pow(2).mean(dim=1, keepdim=True)
    diff_patch = F.avg_pool2d(diff_sq, patch_size, stride=patch_size)
    return torch.exp(-beta * diff_patch)


# ============================================================
# Main FGSR inference
# ============================================================
@torch.no_grad()
def fgsr_inpaint(
    target,
    draft,
    z_init: torch.Tensor,
    cond_z: torch.Tensor,
    mask_z: torch.Tensor,
    scheduler: DDPMSchedule,
    num_inference_steps: int = 50,
    K: int = 3,
    patch_size: int = 4,
    t_spec_start_norm: float = 0.7,
    beta: float = 10.0,
    tol_low: float = 0.05,
    tol_high: float = 0.5,
    boundary_weight: float = 1.0,
    mask_interior_weight: float = 0.0,
    uniform_saliency: bool = False,
    dwt=None,
    verbose: bool = False,
    guidance_scale: float = 1.0,
    cond_emb=None,
    uncond_emb=None,
    known_z=None,
    blend_known: bool = True,
    x0_threshold: float = None,
    k_switch_threshold: float = 0.6,
    spec1_below_tnorm: float = 0.0,
    log_diagnostics: bool = False,
    blend_temperature: float = None,
    x0_thr_strict: float = None,
    x0_thr_loose:  float = None,
    x0_strict_center: float = None,
    x0_strict_width:  float = 0.12,
    drift_k_switch_threshold: float = None,
    # === NEW: usage-map collection for qualitative figures ===
    # When True, the time-averaged soft-blend weight (or hard accept indicator)
    # over Phase-2 verified timesteps is returned in stats["usage_map"].
    return_usage_map: bool = False,
):
    """
    Returns:
        z_final, stats
        If return_usage_map=True, stats["usage_map"] has shape
        [B, 1, H_lat, W_lat] with values in [0, 1].
    """
    def _blend(z_now, t_next_int):
        if not (blend_known and known_z is not None and t_next_int >= 0):
            return z_now
        B_ = z_now.shape[0]
        eps_noise = torch.randn_like(known_z)
        z_known_t = scheduler.q_sample(
            known_z, eps_noise,
            torch.full((B_,), t_next_int, device=z_now.device, dtype=torch.long))
        return mask_z * z_now + (1 - mask_z) * z_known_t

    device = z_init.device
    if dwt is None:
        dwt = DWT2D("haar").to(device)

    ts = scheduler.get_ddim_schedule(num_inference_steps)
    N = len(ts)
    z = z_init.clone()
    B, C, H, W = z.shape

    target_calls = 0
    draft_calls = 0
    total_proposed = 0
    total_accepted = 0
    spec_steps = 0
    stab_steps = 0

    # === NEW: usage-map accumulator ===
    if return_usage_map:
        usage_acc = torch.zeros(B, 1, H, W, device=device)
        usage_count = 0

    i = 0
    while i < N:
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if (i + 1) < N else -1
        t_norm = t_cur / (scheduler.num_train_timesteps - 1)
        t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

        # ---------------- Phase 1: stabilization ----------------
        if t_norm > t_spec_start_norm:
            eps_t = target.predict_eps(
                z, t_tensor, cond_z, mask_z,
                cond_emb=cond_emb, uncond_emb=uncond_emb,
                guidance_scale=guidance_scale,
            )
            target_calls += 1
            z, _ = scheduler.ddim_step(z, eps_t, t_cur, t_prev)
            z = _blend(z, t_prev)
            stab_steps += 1
            if verbose:
                print(f"[stab] i={i} t={t_cur} (target only)")
            i += 1
            continue

        # ---------------- Phase 2: speculative refinement ----------------
        spec_steps += 1

        sal = combined_saliency(z, mask_z, dwt,
                                boundary_weight=boundary_weight,
                                interior_weight=mask_interior_weight,
                                target_size=(H, W),
                                uniform=uniform_saliency)
        sal_patch = F.avg_pool2d(sal, patch_size, stride=patch_size)
        tol_patch = tol_low + (tol_high - tol_low) * (1 - sal_patch)

        eps_t = target.predict_eps(
            z, t_tensor, cond_z, mask_z,
            cond_emb=cond_emb, uncond_emb=uncond_emb,
            guidance_scale=guidance_scale,
        )
        target_calls += 1
        eps_d = draft(z, t_tensor, cond_z, mask_z)
        draft_calls += 1

        sa = scheduler.sqrt_alphas_cumprod[t_cur].view(1, 1, 1, 1)
        som = scheduler.sqrt_one_minus_alphas_cumprod[t_cur].view(1, 1, 1, 1)
        x0_t = (z - som * eps_t) / sa
        x0_d = (z - som * eps_d) / sa
        x0_delta = ((x0_d - x0_t) ** 2).mean(dim=1, keepdim=True)
        x0_delta_patch = F.avg_pool2d(x0_delta, patch_size, stride=patch_size)

        a_patch = patch_agreement(eps_d, eps_t, patch_size, beta)
        accept_eps = (a_patch > (1 - tol_patch))

        if x0_strict_center is not None and x0_thr_strict is not None and x0_thr_loose is not None:
            s = float(math.exp(-((t_norm - x0_strict_center) ** 2)
                               / (2.0 * x0_strict_width ** 2)))
            x0_thr_effective = x0_thr_loose * (1.0 - s) + x0_thr_strict * s
        else:
            x0_thr_effective = x0_threshold

        if x0_thr_effective is not None:
            accept_x0 = (x0_delta_patch < x0_thr_effective)
            accept_patch = (accept_eps & accept_x0).float()
        else:
            accept_patch = accept_eps.float()

        mask_patch = F.max_pool2d(mask_z, patch_size, stride=patch_size)
        accept_full = F.interpolate(accept_patch, size=(H, W), mode="nearest")
        mask_full = F.interpolate(mask_patch, size=(H, W), mode="nearest")
        accept_full = accept_full * (mask_full >= 0.5).float()

        mask_patch_bool = (mask_patch >= 0.5).float()
        n_inner = mask_patch_bool.sum().item()
        n_acc_inner = (accept_patch * mask_patch_bool).sum().item()
        total_proposed += n_inner
        total_accepted += n_acc_inner
        accept_rate = n_acc_inner / max(n_inner, 1)

        if blend_temperature is not None and blend_temperature > 0:
            margin_patch = a_patch - (1.0 - tol_patch)
            w_patch = torch.sigmoid(margin_patch / blend_temperature)
            if x0_thr_effective is not None:
                x0_margin = (x0_thr_effective - x0_delta_patch) / max(blend_temperature, 1e-6)
                w_patch = w_patch * torch.sigmoid(x0_margin)
            w_full = F.interpolate(w_patch, size=(H, W), mode="nearest")
            w_full = w_full * (mask_full >= 0.5).float()
            eps_mix = eps_t + w_full * (eps_d - eps_t)
            # === NEW: accumulate soft weight ===
            if return_usage_map:
                usage_acc = usage_acc + w_full.detach()
                usage_count = usage_count + 1
        else:
            eps_mix = accept_full * eps_d + (1 - accept_full) * eps_t
            # === NEW: accumulate hard accept indicator ===
            if return_usage_map:
                usage_acc = usage_acc + accept_full.detach()
                usage_count = usage_count + 1

        z_next, _ = scheduler.ddim_step(z, eps_mix, t_cur, t_prev)
        z_next = _blend(z_next, t_prev)

        x0_drift = x0_delta.mean().item()

        if log_diagnostics:
            eps_drift = (eps_d - eps_t).pow(2).mean().item()
            print(f"  [diag] t={t_cur:4d}  t_norm={t_norm:.3f}  "
                  f"accept={accept_rate:.3f}  "
                  f"eps_drift={eps_drift:.5f}  x0_drift={x0_drift:.5f}  "
                  f"sal_mean={sal.mean().item():.3f}")

        drift_ok = True
        if drift_k_switch_threshold is not None:
            drift_ok = (x0_drift < drift_k_switch_threshold)

        use_specK = (
            (accept_rate > k_switch_threshold)
            and drift_ok
            and (K > 1)
            and ((i + K) <= N)
            and (t_norm >= spec1_below_tnorm)
        )
        if use_specK:
            z = z_next
            for k in range(1, K):
                if (i + k + 1) > N: break
                t_cur_k = int(ts[i + k].item())
                t_prev_k = int(ts[i + k + 1].item()) if (i + k + 1) < N else -1
                t_tensor_k = torch.full((B,), t_cur_k, device=device, dtype=torch.long)
                eps_d_k = draft(z, t_tensor_k, cond_z, mask_z)
                draft_calls += 1
                z_step, _ = scheduler.ddim_step(z, eps_d_k, t_cur_k, t_prev_k)
                z = _blend(z_step, t_prev_k)
            if verbose:
                print(f"[spec-K] i={i}..{i+K-1} t={t_cur} "
                      f"accept={accept_rate:.2f}")
            i += K
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

    # === NEW: append usage map to stats ===
    if return_usage_map:
        if usage_count > 0:
            usage_map = (usage_acc / usage_count).clamp(0.0, 1.0)
        else:
            usage_map = torch.zeros(B, 1, H, W, device=device)
        stats["usage_map"] = usage_map.cpu()
        stats["usage_count"] = usage_count

    return z, stats


# ============================================================
# Baseline: target-only DDIM with optional CFG and mask-blending
# ============================================================
@torch.no_grad()
def baseline_inpaint(target, z_init, cond_z, mask_z, scheduler,
                     num_inference_steps=50, guidance_scale=1.0,
                     cond_emb=None, uncond_emb=None,
                     known_z=None, blend_known=True):
    ts = scheduler.get_ddim_schedule(num_inference_steps)
    N = len(ts)
    z = z_init.clone()
    B = z.shape[0]
    for i in range(N):
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if (i + 1) < N else -1
        t_tensor = torch.full((B,), t_cur, device=z.device, dtype=torch.long)

        eps_t = target.predict_eps(
            z, t_tensor, cond_z, mask_z,
            cond_emb=cond_emb, uncond_emb=uncond_emb,
            guidance_scale=guidance_scale,
        )
        z, _ = scheduler.ddim_step(z, eps_t, t_cur, t_prev)

        if blend_known and known_z is not None and t_prev >= 0:
            eps_noise = torch.randn_like(known_z)
            z_known_t = scheduler.q_sample(known_z, eps_noise,
                                           torch.full((B,), t_prev, device=z.device,
                                                      dtype=torch.long))
            z = mask_z * z + (1 - mask_z) * z_known_t

    return z, {"target_calls": N, "nfe_total": N, "accept_rate": None}
