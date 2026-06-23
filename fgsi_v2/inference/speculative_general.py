"""
Task-agnostic Frequency-Guided Speculative Refinement (general FreqSpec).

This is the AAAI-track generalization of `speculative.fgsr_inpaint`. The WACV
inpainting sampler is left untouched; this file re-expresses the same five
calibration mechanisms over a *region* abstraction so the identical verifier
serves multiple conditional-diffusion tasks:

    region_z  = mask        (inpainting)        -> substitution inside the hole
    region_z  = all-ones    (super-resolution)  -> the whole field is verified

The acceptance rule, soft blending, predicted-x0 gate, timestep-dependent
strictness, and drift-aware K-step gating are all defined per patch over the
region and are therefore task-independent. Task-specific input assembly
(masked-image latent vs. low-res RGB, noise-level conditioning, known-region
blending) is absorbed by the target/draft wrappers, not by this sampler.

Interface contract (both tasks):
    target.predict_eps(z_t, t, cond_z, region_z,
                       cond_emb=..., uncond_emb=..., guidance_scale=...)
        -> eps_hat   [B, 4, H, W]
    draft(z_t, t, cond_z, region_z) -> eps_hat  [B, 4, H, W]
        (draft may ignore region_z when the task has no mask channel)
"""
import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.wavelet import DWT2D, combined_saliency, lwd_wavelet_saliency
from training.scheduler import DDPMSchedule
from freqspec_core.verifier import VerifierConfig, verify_step


def patch_agreement(eps_a, eps_b, patch_size: int, beta: float):
    """Per-patch agreement: s = exp(-beta * mean_channel ||eps_a - eps_b||^2)."""
    diff_sq = (eps_a - eps_b).pow(2).mean(dim=1, keepdim=True)
    diff_patch = F.avg_pool2d(diff_sq, patch_size, stride=patch_size)
    return torch.exp(-beta * diff_patch)


@torch.no_grad()
def fgsr_refine(
    target,
    draft,
    z_init: torch.Tensor,
    cond_z: torch.Tensor,
    region_z: torch.Tensor,
    scheduler: DDPMSchedule,
    num_inference_steps: int = 50,
    exact_schedule: bool = False,
    K: int = 3,
    patch_size: int = 4,
    t_spec_start_norm: float = 0.7,
    beta: float = 10.0,
    tol_low: float = 0.05,
    tol_high: float = 0.5,
    # saliency configuration. For SR pass boundary_weight=0, interior_weight=0
    # so the strictness prior is the pure wavelet high-frequency map (LWD), which
    # is the cleanest "frequency-guided" instantiation (SR == HF recovery).
    boundary_weight: float = 0.0,
    mask_interior_weight: float = 0.0,
    uniform_saliency: bool = False,
    saliency_signal: str = "wavelet",
    saliency_use_base: bool = True,
    dwt=None,
    verbose: bool = False,
    guidance_scale: float = 1.0,
    cond_emb=None,
    uncond_emb=None,
    # known-region re-injection. Inpainting passes known_z=z0, blend_known=True.
    # SR leaves known_z=None (whole field is generated; the LR anchor enters
    # through the target/draft conditioning, not through blending).
    known_z=None,
    blend_known: bool = False,
    # calibration knobs (identical semantics to the WACV sampler)
    x0_threshold: float = None,
    k_switch_threshold: float = 0.6,
    spec1_below_tnorm: float = 0.0,
    log_diagnostics: bool = False,
    blend_temperature: float = None,
    x0_thr_strict: float = None,
    x0_thr_loose: float = None,
    x0_strict_center: float = None,
    x0_strict_width: float = 0.12,
    saliency_x0_coupling: float = 0.0,
    drift_k_switch_threshold: float = None,
    return_usage_map: bool = False,
    collect_patch_logs: bool = False,
    # extra kwargs forwarded verbatim to target.predict_eps (e.g. noise_level
    # for the SR upscaler). Kept opaque so the sampler stays task-agnostic.
    target_extra: dict = None,
):
    """Generalized FreqSpec sampler. Returns (z_final, stats)."""
    target_extra = target_extra or {}

    def _blend(z_now, t_next_int):
        if not (blend_known and known_z is not None and t_next_int >= 0):
            return z_now
        B_ = z_now.shape[0]
        eps_noise = torch.randn_like(known_z)
        z_known_t = scheduler.q_sample(
            known_z, eps_noise,
            torch.full((B_,), t_next_int, device=z_now.device, dtype=torch.long))
        return region_z * z_now + (1 - region_z) * z_known_t

    device = z_init.device
    if dwt is None:
        dwt = DWT2D("haar").to(device)

    ts = (scheduler.get_ddim_schedule_exact(num_inference_steps)
          if exact_schedule else scheduler.get_ddim_schedule(num_inference_steps))
    N = len(ts)
    z = z_init.clone()
    B, C, H, W = z.shape

    target_calls = 0
    draft_calls = 0
    total_proposed = 0
    total_accepted = 0
    spec_steps = 0
    stab_steps = 0

    if return_usage_map:
        usage_acc = torch.zeros(B, 1, H, W, device=device)
        usage_count = 0
    if collect_patch_logs:
        log_chunks = {"d_x0": [], "s_eps": [], "w": [], "saliency": [],
                      "wav": [], "t_norm": []}

    # Per-step decision is delegated to the task-agnostic verifier core.
    vcfg = VerifierConfig(
        patch_size=patch_size, beta=beta, tol_low=tol_low, tol_high=tol_high,
        blend_temperature=blend_temperature, x0_threshold=x0_threshold,
        x0_thr_strict=x0_thr_strict, x0_thr_loose=x0_thr_loose,
        x0_strict_center=x0_strict_center, x0_strict_width=x0_strict_width,
        saliency_x0_coupling=saliency_x0_coupling,
    )

    i = 0
    while i < N:
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if (i + 1) < N else -1
        t_norm = t_cur / (scheduler.num_train_timesteps - 1)
        t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

        # ---------------- Phase 1: target-only stabilization ----------------
        if t_norm > t_spec_start_norm:
            eps_t = target.predict_eps(
                z, t_tensor, cond_z, region_z,
                cond_emb=cond_emb, uncond_emb=uncond_emb,
                guidance_scale=guidance_scale, **target_extra,
            )
            target_calls += 1
            z, _ = scheduler.ddim_step(z, eps_t, t_cur, t_prev)
            z = _blend(z, t_prev)
            stab_steps += 1
            i += 1
            continue

        # ---------------- Phase 2: speculative refinement ----------------
        spec_steps += 1

        sal = combined_saliency(z, region_z, dwt,
                                boundary_weight=boundary_weight,
                                interior_weight=mask_interior_weight,
                                target_size=(H, W),
                                uniform=uniform_saliency,
                                signal=saliency_signal,
                                use_base_signal=saliency_use_base)
        sal_patch = F.avg_pool2d(sal, patch_size, stride=patch_size)

        eps_t = target.predict_eps(
            z, t_tensor, cond_z, region_z,
            cond_emb=cond_emb, uncond_emb=uncond_emb,
            guidance_scale=guidance_scale, **target_extra,
        )
        target_calls += 1
        eps_d = draft(z, t_tensor, cond_z, region_z)
        draft_calls += 1

        sa = scheduler.sqrt_alphas_cumprod[t_cur].view(1, 1, 1, 1)
        som = scheduler.sqrt_one_minus_alphas_cumprod[t_cur].view(1, 1, 1, 1)
        x0_t = (z - som * eps_t) / sa
        x0_d = (z - som * eps_d) / sa

        region_patch = F.max_pool2d(region_z, patch_size, stride=patch_size)
        region_full = F.interpolate(region_patch, size=(H, W), mode="nearest")

        # ----- task-agnostic verification (identical for inpaint and SR) -----
        out = verify_step(eps_d, eps_t, x0_d, x0_t, sal_patch,
                          region_patch, t_norm, vcfg, region_full=region_full)
        eps_mix = out["eps_mix"]
        accept_patch = out["accept_patch"]
        accept_rate = out["accept_rate"]
        w_full = out["w_full"]

        region_patch_bool = (region_patch >= 0.5).float()
        total_proposed += region_patch_bool.sum().item()
        total_accepted += (accept_patch * region_patch_bool).sum().item()

        if return_usage_map:
            # w_full == soft weight (blend mode) or accept indicator (hard mode)
            usage_acc = usage_acc + w_full.detach()
            usage_count = usage_count + 1

        if collect_patch_logs:
            sel = region_patch_bool.bool().view(-1)
            a_wav = lwd_wavelet_saliency(z, dwt, target_size=(H, W))
            a_wav_patch = F.avg_pool2d(a_wav, patch_size, stride=patch_size)
            log_chunks["d_x0"].append(out["d_x0"].detach().float().view(-1)[sel].cpu())
            log_chunks["s_eps"].append(out["s_eps"].detach().float().view(-1)[sel].cpu())
            log_chunks["w"].append(out["w_patch"].detach().float().view(-1)[sel].cpu())
            log_chunks["saliency"].append(sal_patch.detach().float().view(-1)[sel].cpu())
            log_chunks["wav"].append(a_wav_patch.detach().float().view(-1)[sel].cpu())
            n_sel = int(sel.sum().item())
            log_chunks["t_norm"].append(torch.full((n_sel,), float(t_norm), dtype=torch.float32))

        z_next, _ = scheduler.ddim_step(z, eps_mix, t_cur, t_prev)
        z_next = _blend(z_next, t_prev)

        x0_drift = out["d_x0"].mean().item()
        if log_diagnostics:
            eps_drift = (eps_d - eps_t).pow(2).mean().item()
            print(f"  [diag] t={t_cur:4d} t_norm={t_norm:.3f} accept={accept_rate:.3f} "
                  f"eps_drift={eps_drift:.5f} x0_drift={x0_drift:.5f} "
                  f"sal_mean={sal.mean().item():.3f}")

        drift_ok = True
        if drift_k_switch_threshold is not None:
            drift_ok = (x0_drift < drift_k_switch_threshold)

        use_specK = (
            (accept_rate > k_switch_threshold) and drift_ok and (K > 1)
            and ((i + K) <= N) and (t_norm >= spec1_below_tnorm)
        )
        if use_specK:
            z = z_next
            for k in range(1, K):
                if (i + k + 1) > N:
                    break
                t_cur_k = int(ts[i + k].item())
                t_prev_k = int(ts[i + k + 1].item()) if (i + k + 1) < N else -1
                t_tensor_k = torch.full((B,), t_cur_k, device=device, dtype=torch.long)
                eps_d_k = draft(z, t_tensor_k, cond_z, region_z)
                draft_calls += 1
                z_step, _ = scheduler.ddim_step(z, eps_d_k, t_cur_k, t_prev_k)
                z = _blend(z_step, t_prev_k)
            i += K
        else:
            z = z_next
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
    if return_usage_map:
        usage_map = (usage_acc / usage_count).clamp(0.0, 1.0) if usage_count > 0 \
            else torch.zeros(B, 1, H, W, device=device)
        stats["usage_map"] = usage_map.cpu()
        stats["usage_count"] = usage_count
    if collect_patch_logs:
        if len(log_chunks["d_x0"]) > 0:
            stats["patch_logs"] = {k: torch.cat(v, dim=0) for k, v in log_chunks.items()}
        else:
            stats["patch_logs"] = {k: torch.zeros(0, dtype=torch.float32) for k in log_chunks}
    return z, stats


@torch.no_grad()
def baseline_refine(target, z_init, cond_z, region_z, scheduler,
                    num_inference_steps=50, exact_schedule=False, guidance_scale=1.0,
                    cond_emb=None, uncond_emb=None,
                    known_z=None, blend_known=False, target_extra=None):
    """Target-only DDIM baseline (task-agnostic)."""
    target_extra = target_extra or {}
    ts = (scheduler.get_ddim_schedule_exact(num_inference_steps)
          if exact_schedule else scheduler.get_ddim_schedule(num_inference_steps))
    N = len(ts)
    z = z_init.clone()
    B = z.shape[0]
    for i in range(N):
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if (i + 1) < N else -1
        t_tensor = torch.full((B,), t_cur, device=z.device, dtype=torch.long)
        eps_t = target.predict_eps(
            z, t_tensor, cond_z, region_z,
            cond_emb=cond_emb, uncond_emb=uncond_emb,
            guidance_scale=guidance_scale, **target_extra,
        )
        z, _ = scheduler.ddim_step(z, eps_t, t_cur, t_prev)
        if blend_known and known_z is not None and t_prev >= 0:
            eps_noise = torch.randn_like(known_z)
            z_known_t = scheduler.q_sample(
                known_z, eps_noise,
                torch.full((B,), t_prev, device=z.device, dtype=torch.long))
            z = region_z * z + (1 - region_z) * z_known_t
    return z, {"target_calls": N, "nfe_total": N, "accept_rate": None}