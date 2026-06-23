"""
freqspec_core.verifier — task-agnostic per-step speculative verifier.

This is the AAAI Phase-1 deliverable: the FreqSpec acceptance logic extracted
into pure tensor functions that know nothing about inpainting, super-resolution,
the model backbone, or the sampler schedule. A task supplies already-computed
draft/target predictions, a saliency prior, and a region; the verifier returns
the per-patch decision and the blended prediction.

The four primitives the task code composes (one verify call per task):

    compute_agreement(...)   -> (s_eps, d_x0)      # epsilon + predicted-x0 signals
    compute_acceptance(...)  -> (accept, w)        # saliency-modulated decision
    blend_predictions(...)   -> eps_mix            # soft per-patch blend
    compute_target_usage(...)-> accept_rate        # region-restricted accept rate

`verify_step(...)` composes all four so a task reads:

    out = verify_step(eps_draft, eps_target, x0_draft, x0_target,
                      saliency_patch, region_patch, t_norm, cfg)
    eps_mix, accept_rate = out["eps_mix"], out["accept_rate"]

The same verify_step is used for inpainting (region = mask) and super-resolution
(region = ones) with no task-specific acceptance branch — this is the property
the AAAI paper claims.

NOTE on scope: this module owns the *per-step* decision. The two-phase schedule
and drift-aware K-step lookahead are trajectory-level controls and remain in the
sampler loop; the sampler calls these primitives at each verified timestep.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================
@dataclass
class VerifierConfig:
    """Per-step verifier hyperparameters (all task-agnostic)."""
    patch_size: int = 4
    beta: float = 10.0                 # epsilon-agreement temperature (Eq. 8)
    tol_low: float = 0.03              # tolerance at high-saliency patches (Eq. 7)
    tol_high: float = 0.30             # tolerance at low-saliency patches (Eq. 7)
    blend_temperature: float = 0.10    # soft-blend temperature (Eq. 13); 0 -> hard
    # predicted-x0 gate (Eq. 12). Either a static threshold, or a
    # timestep-dependent Gaussian strict/loose interpolation (Eq. 15).
    x0_threshold: float = None
    x0_thr_strict: float = None
    x0_thr_loose: float = None
    x0_strict_center: float = None
    x0_strict_width: float = 0.12
    # SR frequency ablation: couple the (wavelet) saliency into the binding x0
    # gate so high-frequency patches get a tighter threshold. 0 = current
    # behaviour (x0 gate independent of saliency); >0 injects frequency into the
    # acceptance signal, the deployable form of the AURC frequency-mixing result.
    saliency_x0_coupling: float = 0.0


# ============================================================
# Primitive 1: agreement signals
# ============================================================
def compute_agreement(eps_draft, eps_target, x0_draft, x0_target,
                      patch_size: int, beta: float):
    """Per-patch draft/target agreement signals.

    Returns:
        s_eps [B,1,h,w]: epsilon-space agreement, exp(-beta * mean ||deps||^2),
                         in (0,1]; -> 1 as predictions match (Eq. 8).
        d_x0  [B,1,h,w]: predicted-clean-latent disagreement, mean ||dx0||^2
                         (Eq. 11). The quantity the decoder actually sees.
    """
    deps2 = (eps_draft - eps_target).pow(2).mean(dim=1, keepdim=True)
    s_eps = torch.exp(-beta * F.avg_pool2d(deps2, patch_size, stride=patch_size))
    dx02 = (x0_draft - x0_target).pow(2).mean(dim=1, keepdim=True)
    d_x0 = F.avg_pool2d(dx02, patch_size, stride=patch_size)
    return s_eps, d_x0


def saliency_modulated_tolerance(saliency_patch, tol_low: float, tol_high: float):
    """Eq. 7: high saliency -> tol_low (strict); low saliency -> tol_high."""
    return tol_low + (tol_high - tol_low) * (1.0 - saliency_patch)


def resolve_x0_threshold(t_norm: float, cfg: VerifierConfig):
    """Timestep-dependent x0 threshold (Eq. 15) or a static threshold."""
    if (cfg.x0_strict_center is not None and cfg.x0_thr_strict is not None
            and cfg.x0_thr_loose is not None):
        s = math.exp(-((t_norm - cfg.x0_strict_center) ** 2)
                     / (2.0 * cfg.x0_strict_width ** 2))
        return cfg.x0_thr_loose * (1.0 - s) + cfg.x0_thr_strict * s
    return cfg.x0_threshold


# ============================================================
# Primitive 2: acceptance
# ============================================================
def compute_acceptance(s_eps, d_x0, saliency_patch, region_patch, t_norm,
                       cfg: VerifierConfig):
    """Saliency-modulated acceptance over a region.

    Combines the epsilon-agreement test (Eq. 9), the predicted-x0 gate (Eq. 12),
    and the soft-blend weight (Eq. 13). Restricted to region patches; patches
    outside the region get accept = w = 0 (handled by the caller's region mask).

    Returns:
        accept [B,1,h,w] in {0,1}: hard accept indicator (for usage / logging).
        w      [B,1,h,w] in [0,1]: soft-blend weight (== accept if hard mode).
    """
    tol_patch = saliency_modulated_tolerance(saliency_patch, cfg.tol_low, cfg.tol_high)
    region_bool = (region_patch >= 0.5).float()

    accept_eps = (s_eps > (1.0 - tol_patch))
    x0_thr = resolve_x0_threshold(t_norm, cfg)
    # Couple (wavelet) saliency into the x0 gate: high-saliency patches get a
    # tighter effective threshold. x0_thr_eff is a tensor when coupling>0.
    if x0_thr is not None and cfg.saliency_x0_coupling > 0:
        x0_thr_eff = x0_thr * (1.0 - cfg.saliency_x0_coupling
                               * saliency_patch.clamp(0.0, 1.0))
    else:
        x0_thr_eff = x0_thr
    if x0_thr is not None:
        accept = (accept_eps & (d_x0 < x0_thr_eff)).float() * region_bool
    else:
        accept = accept_eps.float() * region_bool

    if cfg.blend_temperature is not None and cfg.blend_temperature > 0:
        bt = cfg.blend_temperature
        margin = s_eps - (1.0 - tol_patch)
        w = torch.sigmoid(margin / bt)
        if x0_thr is not None:
            w = w * torch.sigmoid((x0_thr_eff - d_x0) / max(bt, 1e-6))
        w = w * region_bool
    else:
        w = accept
    return accept, w


# ============================================================
# Primitive 3: blending
# ============================================================
def blend_predictions(eps_draft, eps_target, w_full):
    """Eq. 14: per-patch soft blend toward the draft. w_full is full latent res."""
    return eps_target + w_full * (eps_draft - eps_target)


# ============================================================
# Primitive 4: target usage
# ============================================================
def compute_target_usage(accept_patch, region_patch):
    """Region-restricted accept rate (fraction of region patches drafted)."""
    region_bool = (region_patch >= 0.5).float()
    n_inner = region_bool.sum().item()
    n_acc = (accept_patch * region_bool).sum().item()
    return n_acc / max(n_inner, 1)


# ============================================================
# Composition: one verify call per task
# ============================================================
def upsample_patch_to(map_patch, H: int, W: int, region_full=None):
    """Nearest-upsample a patch-grid map to full latent res, region-masked."""
    full = F.interpolate(map_patch, size=(H, W), mode="nearest")
    if region_full is not None:
        full = full * (region_full >= 0.5).float()
    return full


def verify_step(eps_draft, eps_target, x0_draft, x0_target,
                saliency_patch, region_patch, t_norm, cfg: VerifierConfig,
                region_full=None):
    """Task-agnostic per-step verification. Composes the four primitives.

    Inputs at patch resolution: saliency_patch, region_patch.
    Inputs at full latent resolution: eps_*, x0_*.
    region_full (optional) is the full-res region used to mask the blend weight;
    if None it is upsampled from region_patch.

    Returns dict with:
        eps_mix      [B,4,H,W]  blended noise prediction
        w_full       [B,1,H,W]  soft-blend weight at full res
        accept_patch [B,1,h,w]  hard accept indicator at patch res
        accept_rate  float      region-restricted accept rate
    """
    B, C, H, W = eps_draft.shape
    s_eps, d_x0 = compute_agreement(eps_draft, eps_target, x0_draft, x0_target,
                                    cfg.patch_size, cfg.beta)
    accept_patch, w_patch = compute_acceptance(
        s_eps, d_x0, saliency_patch, region_patch, t_norm, cfg)
    if region_full is None:
        region_full = upsample_patch_to(region_patch, H, W)
    w_full = upsample_patch_to(w_patch, H, W, region_full)
    eps_mix = blend_predictions(eps_draft, eps_target, w_full)
    accept_rate = compute_target_usage(accept_patch, region_patch)
    return {
        "eps_mix": eps_mix,
        "w_full": w_full,
        "w_patch": w_patch,
        "accept_patch": accept_patch,
        "accept_rate": accept_rate,
        "s_eps": s_eps,
        "d_x0": d_x0,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = VerifierConfig(blend_temperature=0.10, x0_thr_strict=0.02,
                         x0_thr_loose=0.07, x0_strict_center=0.45)
    H = W = 16
    epsd = torch.randn(1, 4, H, W); epst = epsd + 0.01 * torch.randn_like(epsd)
    x0d = torch.randn(1, 4, H, W); x0t = x0d + 0.01 * torch.randn_like(x0d)
    ph = H // cfg.patch_size
    sal = torch.rand(1, 1, ph, ph)
    region = torch.ones(1, 1, ph, ph)
    out = verify_step(epsd, epst, x0d, x0t, sal, region, 0.45, cfg)
    print(f"eps_mix={tuple(out['eps_mix'].shape)} accept_rate={out['accept_rate']:.3f} "
          f"w_full={tuple(out['w_full'].shape)}")