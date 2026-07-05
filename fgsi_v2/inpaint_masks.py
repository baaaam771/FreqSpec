#!/usr/bin/env python
"""
inpaint_masks.py — deterministic inpainting-mask generation.

Masks are a Stage-1 invariant: comparisons across stages are only valid if
the mask for validation image i is bit-identical. Every mask here is derived
from (mask_seed, image_index) alone, so any stage regenerates the same masks
without storing them. Convention: M=1 inside the hole (region to generate),
M=0 on the known region.

Kinds:
    box      : one axis-aligned box, area fraction ~U[amin, amax]
    freeform : brush strokes (random walks with varying radius), redrawn
               until area fraction falls in [amin, amax]
    mixed    : box with probability p_box (default 0.4), else freeform
               (matches the training-mask convention of the FreqSpec line)
"""
import numpy as np
import torch


def _box(rng, H, W, amin, amax):
    m = np.zeros((H, W), np.float32)
    a = rng.uniform(amin, amax)
    ar = np.exp(rng.uniform(-0.5, 0.5))            # aspect ratio
    bh = int(round(np.sqrt(a * H * W * ar)))
    bw = int(round(np.sqrt(a * H * W / ar)))
    bh, bw = min(max(bh, 2), H - 1), min(max(bw, 2), W - 1)
    y0 = rng.integers(0, H - bh + 1)
    x0 = rng.integers(0, W - bw + 1)
    m[y0:y0 + bh, x0:x0 + bw] = 1.0
    return m


def _freeform(rng, H, W, amin, amax, max_tries=20):
    for _ in range(max_tries):
        m = np.zeros((H, W), np.float32)
        n_strokes = rng.integers(1, 5)
        for _ in range(n_strokes):
            y, x = rng.uniform(0, H), rng.uniform(0, W)
            n_seg = rng.integers(4, 16)
            ang = rng.uniform(0, 2 * np.pi)
            for _ in range(n_seg):
                ang += rng.uniform(-0.9, 0.9)
                ln = rng.uniform(H * 0.05, H * 0.25)
                r = rng.uniform(H * 0.03, H * 0.10)
                y2, x2 = y + ln * np.sin(ang), x + ln * np.cos(ang)
                steps = max(int(ln), 1)
                for s in range(steps + 1):
                    cy = y + (y2 - y) * s / steps
                    cx = x + (x2 - x) * s / steps
                    yy, xx = np.ogrid[:H, :W]
                    m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1.0
                y, x = y2, x2
        frac = m.mean()
        if amin <= frac <= amax:
            return m
    return _box(rng, H, W, amin, amax)              # fallback


def make_mask(index, H, W, kind="mixed", seed=0, amin=0.05, amax=0.35,
              p_box=0.4):
    """Deterministic mask for image `index`. Returns float32 [H,W], 1=hole."""
    rng = np.random.default_rng([seed, int(index)])
    if kind == "box" or (kind == "mixed" and rng.random() < p_box):
        return _box(rng, H, W, amin, amax)
    return _freeform(rng, H, W, amin, amax)


def make_mask_batch(indices, H, W, device, **kw):
    """[B,1,H,W] float tensor of masks for the given global indices."""
    ms = [make_mask(i, H, W, **kw) for i in indices]
    return torch.from_numpy(np.stack(ms)[:, None]).to(device)


def token_mask(M, p):
    """Image mask [B,1,H,W] -> token mask [B,N] (True if ANY masked pixel
    overlaps the token's patch) and boundary-band token mask [B,N]
    (dilate - erode on the token grid)."""
    mt = torch.nn.functional.max_pool2d(M, p, stride=p)          # [B,1,h,w]
    dil = torch.nn.functional.max_pool2d(mt, 3, stride=1, padding=1)
    ero = -torch.nn.functional.max_pool2d(-mt, 3, stride=1, padding=1)
    band = (dil - ero).clamp(0, 1)
    return mt.flatten(1) > 0.5, band.flatten(1) > 0.5


if __name__ == "__main__":
    m1 = make_mask(7, 64, 64, "mixed", seed=0)
    m2 = make_mask(7, 64, 64, "mixed", seed=0)
    m3 = make_mask(8, 64, 64, "mixed", seed=0)
    print("deterministic:", (m1 == m2).all(), "| differs by index:",
          (m1 != m3).any(), "| area:", round(float(m1.mean()), 3))
