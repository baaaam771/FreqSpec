#!/usr/bin/env python
"""
metrics_extended.py — Region-aware metrics for reduced-step baseline comparison.

Provides:
  - masked_lpips:   LPIPS computed only inside the mask region
  - boundary_lpips: LPIPS computed only in a narrow band around the mask edge
                    (this is the key metric for demonstrating the boundary-aware
                    saliency contribution)
  - masked_psnr:    PSNR inside the mask region
  - full LPIPS/PSNR/SSIM for reference

All metrics compare against the GROUND-TRUTH image (not the target output),
so they measure restoration quality, not target-preservation.

Usage:
    from metrics_extended import RegionMetrics
    m = RegionMetrics(device="cuda")
    result = m.compute(pred, gt, mask)   # all numpy HWC uint8 or float [0,1]
    # result = {"psnr", "ssim", "lpips", "masked_psnr", "masked_lpips",
    #           "boundary_lpips"}
"""
import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_lib
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import cv2


def _to_float01(img):
    """Convert HWC uint8 or float image to float32 [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def _to_lpips_tensor(img01, device):
    """HWC float[0,1] -> 1x3xHxW tensor in [-1, 1] for LPIPS."""
    t = torch.from_numpy(img01).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def make_boundary_band(mask_bin, k=8):
    """
    Build a boundary band around the mask edge.

    band = dilate(mask, k) XOR erode(mask, k)
    This captures the seam region where blending quality matters most.

    Args:
        mask_bin: HxW binary (0/1) numpy array, 1 = masked (inpaint) region
        k: half-width of the band in pixels

    Returns:
        HxW binary band (1 = inside boundary band)
    """
    mask_u8 = (mask_bin > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    band = (dilated - eroded)  # 1 in the ring around the edge
    return band.astype(np.float32)


class RegionMetrics:
    def __init__(self, device="cuda", lpips_net="alex", boundary_k=8):
        self.device = device
        self.boundary_k = boundary_k
        # AlexNet backbone matches the paper's LPIPS setup
        self.lpips_fn = lpips_lib.LPIPS(net=lpips_net).to(device).eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _lpips_full(self, pred01, gt01):
        p = _to_lpips_tensor(pred01, self.device)
        g = _to_lpips_tensor(gt01, self.device)
        return float(self.lpips_fn(p, g).item())

    @torch.no_grad()
    def _lpips_region(self, pred01, gt01, region_mask):
        """
        LPIPS restricted to a region.

        Strategy: composite both pred and gt so that OUTSIDE the region they
        are identical (we copy gt into both). Then LPIPS differences arise
        only from inside the region. This is a practical region-restricted
        LPIPS; for very small regions it is approximate but monotonic and
        consistent across methods (which is what matters for comparison).
        """
        region = region_mask[..., None]  # HxWx1
        # both images share gt outside the region -> zero contribution there
        pred_comp = pred01 * region + gt01 * (1.0 - region)
        gt_comp = gt01  # gt everywhere
        return self._lpips_full(pred_comp.astype(np.float32),
                                gt_comp.astype(np.float32))

    @torch.no_grad()
    def compute(self, pred, gt, mask):
        """
        Args:
            pred: HWC predicted image (uint8 or float[0,1])
            gt:   HWC ground-truth image
            mask: HxW binary, 1 = masked (inpaint) region
        Returns dict of metrics.
        """
        pred01 = _to_float01(pred)
        gt01 = _to_float01(gt)
        mask_bin = (mask > 0.5).astype(np.float32)
        if mask_bin.ndim == 3:
            mask_bin = mask_bin[..., 0]

        # --- Full-image metrics ---
        psnr_full = sk_psnr(gt01, pred01, data_range=1.0)
        ssim_full = sk_ssim(gt01, pred01, channel_axis=2, data_range=1.0)
        lpips_full = self._lpips_full(pred01, gt01)

        # --- Masked-region metrics ---
        m3 = mask_bin[..., None]
        # masked PSNR: only over masked pixels
        diff2 = ((pred01 - gt01) ** 2) * m3
        denom = m3.sum() * 3 + 1e-8
        mse_masked = diff2.sum() / denom
        masked_psnr = 10.0 * np.log10(1.0 / (mse_masked + 1e-12))
        masked_lpips = self._lpips_region(pred01, gt01, mask_bin)

        # --- Boundary-band metrics (KEY for boundary-aware claim) ---
        band = make_boundary_band(mask_bin, k=self.boundary_k)
        boundary_lpips = self._lpips_region(pred01, gt01, band)

        return {
            "psnr": float(psnr_full),
            "ssim": float(ssim_full),
            "lpips": float(lpips_full),
            "masked_psnr": float(masked_psnr),
            "masked_lpips": float(masked_lpips),
            "boundary_lpips": float(boundary_lpips),
        }


if __name__ == "__main__":
    # quick self-test with random data
    m = RegionMetrics(device="cuda" if torch.cuda.is_available() else "cpu")
    H = W = 256
    gt = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    pred = gt.copy()
    pred[80:160, 80:160] = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[80:160, 80:160] = 1.0
    print(m.compute(pred, gt, mask))
