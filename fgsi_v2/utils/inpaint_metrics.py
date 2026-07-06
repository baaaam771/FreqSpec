#!/usr/bin/env python
"""
utils/inpaint_metrics.py — evaluation helpers for the DiT-inpainting sweep.

review items 7/8/12:
  - region-restricted PSNR (mask, known) and SSIM (known)
  - LPIPS confined to a region (mask, boundary ring) — the caller supplies the
    lpips module (or None)
  - mask-size bucketing (small / medium / large by hole coverage)

All images are in [-1, 1], masks in {0,1}, mask=1 = hole. Metric NAMES follow
the review: quantities compared against the dense-50 output are suffixed
_to_dense50 (they are output-vs-output distances, NOT true trajectory metrics).
"""
import torch
import torch.nn.functional as F


def region_mse(a, b, region):
    """Per-sample mean squared error over `region` (pixels), [-1,1] scale.
    Returns [B]."""
    num = ((a - b) ** 2 * region).flatten(1).sum(1)
    den = region.flatten(1).sum(1) * a.shape[1] + 1e-8
    return num / den


def region_psnr(a, b, region):
    """Per-sample PSNR over `region` (dynamic range 2.0). Returns [B].
    Perfectly-preserved regions saturate; caller may clamp for display."""
    mse = region_mse(a, b, region).clamp(min=1e-12)
    return 10 * torch.log10(4.0 / mse)


def _gaussian_window(ch, ksize=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ksize, device=device).float() - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(ch, 1, ksize, ksize).contiguous()
    return w


def ssim_map(a, b, ch, device):
    """Per-pixel SSIM map for images in [-1,1] (rescaled to [0,1]). [B,1,H,W]."""
    a = (a + 1) / 2
    b = (b + 1) / 2
    w = _gaussian_window(ch, device=device)
    pad = w.shape[-1] // 2
    mu_a = F.conv2d(a, w, padding=pad, groups=ch)
    mu_b = F.conv2d(b, w, padding=pad, groups=ch)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    s_a = F.conv2d(a * a, w, padding=pad, groups=ch) - mu_a2
    s_b = F.conv2d(b * b, w, padding=pad, groups=ch) - mu_b2
    s_ab = F.conv2d(a * b, w, padding=pad, groups=ch) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_ab + c1) * (2 * s_ab + c2)) / \
           ((mu_a2 + mu_b2 + c1) * (s_a + s_b + c2))
    return smap.mean(1, keepdim=True)


def region_ssim(a, b, region):
    """Mean SSIM over `region`. Returns [B]."""
    ch = a.shape[1]
    smap = ssim_map(a, b, ch, a.device)
    num = (smap * region).flatten(1).sum(1)
    den = region.flatten(1).sum(1) + 1e-8
    return num / den


def region_lpips(lpips_fn, a, b, region):
    """LPIPS on images with everything outside `region` blanked to the same
    constant in both, so only the region contributes. Returns [B]. If
    lpips_fn is None returns None."""
    if lpips_fn is None:
        return None
    fill = -1.0
    aa = a * region + fill * (1 - region)
    bb = b * region + fill * (1 - region)
    return lpips_fn(aa, bb).flatten()


def mask_size_bucket(coverage, small=0.10, large=0.25):
    """coverage [B] hole fraction -> list of 'small'/'medium'/'large'."""
    out = []
    for c in coverage.tolist():
        out.append("small" if c < small else ("large" if c >= large else "medium"))
    return out
