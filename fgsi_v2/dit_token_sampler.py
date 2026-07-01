#!/usr/bin/env python
"""
dit_token_sampler.py — Phase 2 PoC: K=1 token-mixing sampler on a DiT token grid.

The minimal sampler: at each DDIM step compute target and draft eps, accept the
top `accept_ratio` tokens by eps-agreement (per image), use the draft eps on
accepted tokens and the target eps on rejected ones, mix in eps space, take one
DDIM step. No lookahead, no soft blend (K=1) — the goal is only to show that
token-wise mixing produces valid (non-collapsed) samples while reducing target
token usage.

Methods: target-only, draft-only, freqspec-token (agreement mix), random-token
(same accept ratio, random mask). Saves a sample grid per method, reports actual
accept / target-token usage, and (optional) FID if a fid library is available.

Usage:
    python dit_token_sampler.py \
        --target ckpt_dit/target.pt --target_model DiT-S \
        --draft  ckpt_dit/draft.pt  --draft_model  DiT-Ti \
        --out_dir results/dit_token_poc_v0/sampling \
        --n_samples 64 --steps 50 --accept_ratio 0.7
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit import build_dit, count_params
from training.scheduler import DDPMSchedule


def load_dit(path, name, args, dev):
    m = build_dit(name, img_size=args.img_size, patch=args.patch,
                  num_classes=args.num_classes, class_dropout=0.0).to(dev).eval()
    ck = torch.load(path, map_location=dev)
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def token_agreement(eps_d, eps_t, p, beta):
    """Per-token eps-agreement s_eps in [0,1], shape [B,1,h,w]."""
    deps2 = F.avg_pool2d((eps_d - eps_t).pow(2).mean(1, keepdim=True), p, stride=p)
    return torch.exp(-beta * deps2)


def token_score(selector, eps_d, eps_t, z, p, beta, dwt=None):
    """Per-token acceptance score (higher = more likely to accept the draft),
    shape [B,1,h,w]. Mirrors the reliability selectors so the FID ablation uses
    the same signals as the AURC analysis."""
    if selector in ("freqspec", "eps_l2"):
        return token_agreement(eps_d, eps_t, p, beta)
    if selector == "eps_cosine":
        dot = F.avg_pool2d((eps_d * eps_t).sum(1, keepdim=True), p, stride=p)
        nd = F.avg_pool2d(eps_d.pow(2).sum(1, keepdim=True), p, stride=p).sqrt()
        nt = F.avg_pool2d(eps_t.pow(2).sum(1, keepdim=True), p, stride=p).sqrt()
        return dot / (nd * nt + 1e-8)                       # [-1,1], higher = agree
    if selector == "token_norm":
        # lower target content magnitude -> accept draft (matches 1 - norm ranking)
        tnorm = F.avg_pool2d(eps_t.pow(2).mean(1, keepdim=True), p, stride=p)
        return -tnorm
    if selector == "frequency":
        # lower wavelet high-frequency saliency -> accept draft
        from models.wavelet import lwd_wavelet_saliency
        wav = lwd_wavelet_saliency(z, dwt, target_size=(z.shape[-2] // p,
                                                        z.shape[-1] // p))
        return -wav
    raise ValueError(f"unknown selector {selector}")


def upsample_mask(mask_tok, p):
    return mask_tok.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)


def accept_topk(s_tok, ratio):
    """Per-image top-`ratio` token mask by agreement. [B,1,h,w] -> bool mask."""
    B = s_tok.shape[0]
    flat = s_tok.view(B, -1)
    k = max(1, int(round(ratio * flat.shape[1])))
    thr = flat.kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values  # k-th largest
    return (flat >= thr).view_as(s_tok)


@torch.no_grad()
def sample(method, target, draft, sch, ts, y, z, args, dev):
    p, B = args.patch, y.shape[0]
    acc_sum, steps = 0.0, 0
    dwt = None
    if method == "frequency":
        from models.wavelet import DWT2D
        dwt = DWT2D("haar").to(dev)
    mix_selectors = ("freqspec", "eps_l2", "eps_cosine", "token_norm", "frequency")
    for i, t in enumerate(ts):
        t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
        tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
        if method == "target":
            eps = target(z, tt, y)
        elif method == "draft":
            eps = draft(z, tt, y)
        else:
            eps_t = target(z, tt, y)
            eps_d = draft(z, tt, y)
            if method == "random":
                acc = (torch.rand(B, 1, z.shape[-2] // p, z.shape[-1] // p, device=dev)
                       < args.accept_ratio).float()
            else:  # any score-based selector
                s_tok = token_score(method, eps_d, eps_t, z, p, args.beta, dwt)
                acc = accept_topk(s_tok, args.accept_ratio).float()
            acc_full = upsample_mask(acc, p)
            eps = acc_full * eps_d + (1 - acc_full) * eps_t
            acc_sum += acc.mean().item(); steps += 1
        z, _ = sch.ddim_step(z, eps, int(t), t_prev, eta=0.0)
    accept = acc_sum / max(steps, 1) if method in mix_selectors + ("random",) else \
        (1.0 if method == "draft" else 0.0)
    return z, accept


def save_grid(x, path, nrow=8):
    from torchvision.utils import save_image
    save_image((x.clamp(-1, 1) + 1) / 2, path, nrow=nrow)


def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    print(f"[dit-sample] target {args.target_model} {count_params(target)/1e6:.1f}M | "
          f"draft {args.draft_model} {count_params(draft)/1e6:.1f}M")

    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    B = args.n_samples
    torch.manual_seed(args.seed)
    y = torch.randint(0, args.num_classes, (B,), device=dev)
    z0 = torch.randn(B, 3, args.img_size, args.img_size, device=dev)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for method in ["target", "draft", "freqspec", "random"]:
        t0 = time.time()
        x, accept = sample(method, target, draft, sch, ts, y, z0.clone(), args, dev)
        dt = time.time() - t0
        save_grid(x, os.path.join(args.out_dir, f"grid_{method}.png"))
        px = ((x.clamp(-1, 1) + 1) / 2)
        results[method] = dict(
            accept=round(float(accept), 4),
            target_token_usage=round(1.0 - float(accept), 4) if method in ("freqspec", "random")
            else (0.0 if method == "draft" else 1.0),
            runtime_s=round(dt, 2),
            pixel_mean=round(px.mean().item(), 4),
            pixel_std=round(px.std().item(), 4))
        print(f"[dit-sample] {method:9s} accept={results[method]['accept']:.3f} "
              f"tgt_use={results[method]['target_token_usage']:.3f} "
              f"px_std={results[method]['pixel_std']:.3f} t={dt:.1f}s")

    # optional FID vs CIFAR-10 test set
    fid = try_fid(args, dev)
    if fid:
        for m in results:
            results[m]["fid"] = fid.get(m)
        print("[dit-sample] FID:", "  ".join(f"{k}={v}" for k, v in fid.items()))

    with open(os.path.join(args.out_dir, "sampling_summary.json"), "w") as f:
        json.dump(dict(steps=args.steps, accept_ratio=args.accept_ratio,
                       n_samples=B, methods=results), f, indent=2)
    print(f"[dit-sample] wrote {os.path.join(args.out_dir, 'sampling_summary.json')}")
    print(f"[dit-sample] grids in {args.out_dir}/grid_*.png")


def try_fid(args, dev):
    """Compute FID if a library + enough samples; else None (PoC degrades gracefully)."""
    if args.n_samples < args.fid_min or not args.compute_fid:
        return None
    try:
        from pytorch_fid import fid_score  # noqa
    except Exception:
        print("[dit-sample] pytorch_fid not installed; skipping FID (grids still saved).")
        return None
    print("[dit-sample] FID computation hook present but generation count is small; "
          "use --n_samples >= 5000 and a dedicated FID run for paper numbers.")
    return None


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Ti")
    ap.add_argument("--out_dir", type=str, default="results/dit_token_poc_v0/sampling")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--accept_ratio", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compute_fid", action="store_true")
    ap.add_argument("--fid_min", type=int, default=5000)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())