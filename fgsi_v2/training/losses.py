"""
Wavelet 변환 + LWD-style frequency saliency + boundary-aware extension.

LWD (Sigillo et al. 2025): 
    E(i,j) = (1/C) Σ_c [(z_LH)^2 + (z_HL)^2 + (z_HH)^2]
    A_wavelet ∈ [0,1]^{H×W} (min-max normalized)
    M_t(i,j) = 1 if (A(i,j) + ℓ) ≥ t

Ours (boundary extension for inpainting):
    A_combined = normalize(A_wavelet + λ_b · BoundaryIndicator(mask))
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_WAVELET_COEFFS = {
    "haar": {
        "dec_lo": [1 / math.sqrt(2), 1 / math.sqrt(2)],
        "dec_hi": [-1 / math.sqrt(2), 1 / math.sqrt(2)],
    },
    "db2": {
        "dec_lo": [-0.12940952255092145, 0.22414386804185735,
                   0.836516303737469, 0.48296291314469025],
        "dec_hi": [-0.48296291314469025, 0.836516303737469,
                   -0.22414386804185735, -0.12940952255092145],
    },
}


def _get_dec_filters(wavelet: str):
    try:
        import pywt
        w = pywt.Wavelet(wavelet)
        return list(w.dec_lo), list(w.dec_hi)
    except ImportError:
        if wavelet not in _WAVELET_COEFFS:
            raise ValueError(f"Need pywt for {wavelet}. Built-in: {list(_WAVELET_COEFFS.keys())}")
        c = _WAVELET_COEFFS[wavelet]
        return c["dec_lo"], c["dec_hi"]


class DWT2D(nn.Module):
    """LL, LH, HL, HH decomposition."""
    def __init__(self, wavelet: str = "haar"):
        super().__init__()
        dec_lo_l, dec_hi_l = _get_dec_filters(wavelet)
        dec_lo = torch.tensor(dec_lo_l[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(dec_hi_l[::-1], dtype=torch.float32)
        ll = torch.outer(dec_lo, dec_lo)
        lh = torch.outer(dec_hi, dec_lo)
        hl = torch.outer(dec_lo, dec_hi)
        hh = torch.outer(dec_hi, dec_hi)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)
        self.k = filters.shape[-1]

    def forward(self, x: torch.Tensor):
        # dtype auto-cast: filter를 input dtype에 맞춰서 fp16 학습 호환
        B, C, H, W = x.shape
        f = self.filters.repeat(C, 1, 1, 1)
        if f.dtype != x.dtype:
            f = f.to(x.dtype)
        pad = self.k // 2
        x_p = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y = F.conv2d(x_p, f, stride=2, groups=C)
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


def lwd_wavelet_saliency(latent, dwt, target_size=None, eps=1e-6):
    """LWD Eq.3."""
    _, lh, hl, hh = dwt(latent)
    E = (lh.pow(2) + hl.pow(2) + hh.pow(2)).mean(dim=1, keepdim=True)
    H_out, W_out = target_size or latent.shape[-2:]
    E_up = F.interpolate(E, size=(H_out, W_out), mode="bilinear", align_corners=False)
    B = E_up.shape[0]
    flat = E_up.view(B, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    return ((flat - mn) / (mx - mn + eps)).view(B, 1, H_out, W_out)


def boundary_indicator(mask, kernel=5):
    pad = kernel // 2
    dil = F.max_pool2d(mask, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def _minmax_per_image(x, eps=1e-6):
    B = x.shape[0]
    flat = x.view(B, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    return ((flat - mn) / (mx - mn + eps)).view_as(x)


def _sobel_saliency(latent, target_size, eps=1e-6):
    """Spatial-gradient saliency baseline (Sobel magnitude over channels)."""
    C = latent.shape[1]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=latent.device, dtype=latent.dtype)
    ky = kx.t().contiguous()
    kx = kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    xp = F.pad(latent, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(xp, kx, groups=C)
    gy = F.conv2d(xp, ky, groups=C)
    mag = (gx.pow(2) + gy.pow(2)).mean(dim=1, keepdim=True).sqrt()
    H_out, W_out = target_size or latent.shape[-2:]
    mag = F.interpolate(mag, size=(H_out, W_out), mode="bilinear",
                        align_corners=False)
    return _minmax_per_image(mag, eps)


def _laplacian_saliency(latent, target_size, eps=1e-6):
    """Laplacian energy saliency baseline."""
    C = latent.shape[1]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=latent.device, dtype=latent.dtype)
    k = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    xp = F.pad(latent, (1, 1, 1, 1), mode="reflect")
    lap = F.conv2d(xp, k, groups=C).pow(2).mean(dim=1, keepdim=True)
    H_out, W_out = target_size or latent.shape[-2:]
    lap = F.interpolate(lap, size=(H_out, W_out), mode="bilinear",
                        align_corners=False)
    return _minmax_per_image(lap, eps)


def _variance_saliency(latent, target_size, win=3, eps=1e-6):
    """Local latent-variance saliency baseline (per-window variance)."""
    x = latent.mean(dim=1, keepdim=True)
    pad = win // 2
    mean = F.avg_pool2d(F.pad(x, (pad,)*4, mode="reflect"), win, stride=1)
    mean2 = F.avg_pool2d(F.pad(x.pow(2), (pad,)*4, mode="reflect"), win, stride=1)
    var = (mean2 - mean.pow(2)).clamp_min(0.0)
    H_out, W_out = target_size or latent.shape[-2:]
    var = F.interpolate(var, size=(H_out, W_out), mode="bilinear",
                        align_corners=False)
    return _minmax_per_image(var, eps)


def base_saliency_signal(latent, dwt, signal="wavelet", target_size=None,
                         eps=1e-6, generator=None):
    """Return a base (pre-boundary, pre-interior) saliency map in [0,1].

    signal:
        "wavelet"   - LWD Haar high-frequency energy (paper default)
        "sobel"     - Sobel gradient magnitude
        "laplacian" - Laplacian energy
        "variance"  - local latent variance
        "random"    - uniform random map (control)
        "uniform"   - constant 1.0 (no saliency)
    These are the Table B comparison signals: a frequency-guided method must
    beat sobel / laplacian / variance to justify the wavelet choice.
    """
    H_out, W_out = target_size or latent.shape[-2:]
    if signal == "wavelet":
        return lwd_wavelet_saliency(latent, dwt, target_size=(H_out, W_out), eps=eps)
    if signal == "sobel":
        return _sobel_saliency(latent, (H_out, W_out), eps)
    if signal == "laplacian":
        return _laplacian_saliency(latent, (H_out, W_out), eps)
    if signal == "variance":
        return _variance_saliency(latent, (H_out, W_out), eps=eps)
    if signal == "random":
        r = torch.rand(latent.shape[0], 1, H_out, W_out,
                       device=latent.device, dtype=latent.dtype,
                       generator=generator)
        return r
    if signal in ("uniform", "none"):
        return torch.ones(latent.shape[0], 1, H_out, W_out,
                          device=latent.device, dtype=latent.dtype)
    raise ValueError(f"unknown saliency signal: {signal}")


def combined_saliency(latent, mask, dwt, boundary_weight=1.0,
                     boundary_kernel=5, target_size=None, eps=1e-6,
                     uniform=False, interior_weight=0.0,
                     signal="wavelet", use_base_signal=True):
    """A_combined for inpainting: A_wavelet + λ_b · Boundary + λ_m · Interior.
    
    Args:
        uniform: True이면 wavelet saliency 무시하고 모든 위치에 동일값 (1.0).
                 Boundary도 boundary_weight=0이면 무시됨.
                 둘 다 끄면 saliency = 1.0 everywhere (사실상 unconditional draft use).
        interior_weight: Fix #5 (mask interior penalty). > 0이면 mask 내부 전체를
                 hard region으로 추가 표시. COCO같은 caption-conditioned object
                 completion에서 mask 내부 semantic mismatch를 잡기 위한 신호.
                 Wavelet/boundary는 texture/seam에 강하지만 semantic은 못 잡음.
                 Boundary는 mask 경계만 강조하지만 interior는 mask 전체를 강조.
                 추천 sweep: 0.3 / 0.5 / 0.8 (boundary_weight=1.0과 함께).
    """
    H_out, W_out = target_size or latent.shape[-2:]
    if uniform:
        # Uniform: saliency = 1 everywhere. Wavelet 계산 skip.
        A_w = torch.ones(latent.shape[0], 1, H_out, W_out,
                         device=latent.device, dtype=latent.dtype)
    elif not use_base_signal:
        # Boundary-only / interior-only configs: zero base, components added below.
        A_w = torch.zeros(latent.shape[0], 1, H_out, W_out,
                          device=latent.device, dtype=latent.dtype)
    else:
        A_w = base_saliency_signal(latent, dwt, signal=signal,
                                   target_size=(H_out, W_out), eps=eps)
    if mask.shape[-2:] != (H_out, W_out):
        mask_r = F.interpolate(mask, size=(H_out, W_out), mode="nearest")
    else:
        mask_r = mask
    comb = A_w
    if boundary_weight > 0:
        B_ind = boundary_indicator(mask_r, kernel=boundary_kernel)
        comb = comb + boundary_weight * B_ind
    if interior_weight > 0:
        # Fix #5: mask 내부 전체를 hard region으로 표시.
        # mask_r는 이미 [0,1] 또는 {0,1} 형태로 mask 내부=1, 외부=0.
        # 이걸 그대로 saliency에 더하면 mask 내부 patch의 acceptance threshold가
        # 더 엄격해짐 (saliency 높을수록 tolerance 작음).
        interior_ind = (mask_r >= 0.5).float()
        comb = comb + interior_weight * interior_ind
    B = comb.shape[0]
    flat = comb.view(B, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    return ((flat - mn) / (mx - mn + eps)).view(B, 1, H_out, W_out)


def lwd_time_mask(saliency, t_norm, ell=0.3):
    """
    LWD Eq.6. t_norm: timestep in [0,1] (NOTE: not raw diffusion step).
    For DDPM step in [0,T-1], normalize first: t_norm = t / (T-1).
    """
    t_ = t_norm.view(-1, 1, 1, 1)
    return ((saliency + ell) >= t_).float()


if __name__ == "__main__":
    torch.manual_seed(0)
    dwt = DWT2D("haar")
    z = torch.randn(2, 4, 32, 32)
    m = (torch.rand(2, 1, 32, 32) > 0.7).float()
    A = combined_saliency(z, m, dwt, boundary_weight=1.0)
    t_norm = torch.rand(2)
    M = lwd_time_mask(A, t_norm)
    print(f"A: {A.shape}, range [{A.min():.3f},{A.max():.3f}]")
    print(f"M_t active ratio: {M.mean():.3f}")