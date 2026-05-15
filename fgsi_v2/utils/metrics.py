"""Metrics for FGSI evaluation."""
import torch
import torch.nn.functional as F


def psnr(x, y, max_val=2.0):
    mse = (x - y).pow(2).mean()
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-10))


def ssim(x, y, ws=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x = (x + 1) / 2; y = (y + 1) / 2
    pad = ws // 2
    mu_x = F.avg_pool2d(x, ws, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, ws, stride=1, padding=pad)
    sx = F.avg_pool2d(x * x, ws, stride=1, padding=pad) - mu_x ** 2
    sy = F.avg_pool2d(y * y, ws, stride=1, padding=pad) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, ws, stride=1, padding=pad) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2)
    return (num / den).mean()


def boundary_l1(x, y, mask, kernel=5):
    pad = kernel // 2
    dil = F.max_pool2d(mask, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel, stride=1, padding=pad)
    B = (dil - ero).clamp(0, 1)
    diff = (x - y).abs().mean(dim=1, keepdim=True)
    return (diff * B).sum() / (B.sum() + 1e-6)


def hh_band_psnr(x, y, dwt, max_val=2.0):
    _, _, _, hh_x = dwt(x)
    _, _, _, hh_y = dwt(y)
    mse = (hh_x - hh_y).pow(2).mean()
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-10))


def hole_psnr(x, y, mask, max_val=2.0):
    diff = (x - y).pow(2)
    mse = (diff * mask).sum() / (mask.sum() * x.shape[1] + 1e-6)
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-10))
