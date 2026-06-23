#!/usr/bin/env python
"""
dit_token_analysis.py — FreqSpec verifier on a DiT patch-token grid (Phase 1 PoC).

Goal (one sentence): the same region-wise speculative verifier that operates on a
U-Net spatial grid also operates on a DiT patch-token grid. We take a larger
target DiT and a smaller draft DiT, and at sampled timesteps over a data subset we
compute, per token, the eps-agreement and the predicted-x0 disagreement (the
risk). We then compare token-selection rules by AURC (does the verifier rank safe
tokens well?) and dump a few token accept maps.

Selectors (confidence that the draft is safe on a token; higher = accept first):
    Random            : lower bound
    Eps agreement     : s_eps = exp(-beta * ||deps||^2)         (the core verifier)
    Token-norm        : 1 - normalized target x0 token magnitude (simple DiT prior)
    Frequency-token   : 1 - wavelet HF energy per token         (FreqSpec identity)
Risk = per-token predicted-x0 disagreement d_x0 (what the decoder/sampler sees).

Usage:
    python dit_token_analysis.py \
        --target ckpt_dit/target.pt --target_model DiT-S \
        --draft  ckpt_dit/draft.pt  --draft_model  DiT-Ti \
        --data_root ./data --num_batches 40 --out_dir results/dit_token_poc_v0
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit import build_dit, count_params
from models.wavelet import DWT2D, lwd_wavelet_saliency
from training.scheduler import DDPMSchedule
from analyze_verifier_reliability import risk_coverage_curve, compute_aurc


def load_dit(path, model_name, args, dev):
    m = build_dit(model_name, img_size=args.img_size, patch=args.patch,
                  num_classes=args.num_classes, class_dropout=0.0).to(dev)
    ck = torch.load(path, map_location=dev)
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def get_loader(args):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                       num_workers=args.workers, drop_last=True)


def eps_to_x0(x_t, eps, t, sch):
    sa = sch.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    so = sch.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return (x_t - so * eps) / sa


def token_pool(x, p):
    """Mean over c then avg-pool to token grid: [B,C,H,W] -> [B,1,H/p,W/p]."""
    return F.avg_pool2d(x.pow(2).mean(1, keepdim=True), p, stride=p)


def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    dwt = DWT2D("haar").to(dev)
    p = args.patch
    print(f"[dit-tok] target {args.target_model} {count_params(target)/1e6:.1f}M | "
          f"draft {args.draft_model} {count_params(draft)/1e6:.1f}M | tokens={target.num_tokens}")

    loader = get_loader(args)
    ts_grid = torch.linspace(args.t_min, args.t_max, args.n_timesteps).round().long()

    chunks = {k: [] for k in ("d_x0", "s_eps", "tnorm", "wav")}
    example_maps = []
    nb = 0
    for x, y in loader:
        if nb >= args.num_batches:
            break
        x, y = x.to(dev), y.to(dev)
        for tv in ts_grid:
            t = torch.full((x.shape[0],), int(tv), device=dev, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = sch.q_sample(x, noise, t)
            with torch.no_grad():
                eps_t = target(x_t, t, y)
                eps_d = draft(x_t, t, y)
                x0_t = eps_to_x0(x_t, eps_t, t, sch)
                x0_d = eps_to_x0(x_t, eps_d, t, sch)
            # per-token signals
            deps2 = token_pool(eps_d - eps_t, p)                 # [B,1,h,w]
            s_eps = torch.exp(-args.beta * deps2)
            d_x0 = token_pool(x0_d - x0_t, p)
            tnorm = token_pool(x0_t, p)                          # target content magnitude
            wav = lwd_wavelet_saliency(x_t, dwt, target_size=(x_t.shape[-2] // p,
                                                              x_t.shape[-1] // p))
            for k, v in (("d_x0", d_x0), ("s_eps", s_eps), ("tnorm", tnorm), ("wav", wav)):
                chunks[k].append(v.flatten().cpu().numpy())
            if len(example_maps) < args.n_example_maps and int(tv) == int(ts_grid[len(ts_grid) // 2]):
                example_maps.append(dict(
                    t=int(tv),
                    s_eps=s_eps[0, 0].cpu().numpy(),
                    d_x0=d_x0[0, 0].cpu().numpy(),
                    wav=wav[0, 0].cpu().numpy(),
                    accept=(s_eps[0, 0] > (1 - args.tol)).float().cpu().numpy()))
        nb += 1
        if nb % 5 == 0:
            print(f"[dit-tok] {nb}/{args.num_batches} batches")

    merged = {k: np.concatenate(v) for k, v in chunks.items()}
    n = merged["d_x0"].size
    print(f"[dit-tok] {n} tokens total")

    risk = merged["d_x0"]
    rng = np.random.default_rng(0)
    selectors = {
        "Random":            rng.random(n),
        "Eps agreement":     merged["s_eps"],
        "Token-norm":        1.0 - _norm(merged["tnorm"]),
        "Frequency-token":   1.0 - _norm(merged["wav"]),
    }
    coverages = np.linspace(0.05, 1.0, 40)
    os.makedirs(args.out_dir, exist_ok=True)
    aurc = {}
    for name, conf in selectors.items():
        cov, risks = risk_coverage_curve(conf, risk, coverages)
        aurc[name] = compute_aurc(cov, risks)
    # oracle lower bound
    cov, risks = risk_coverage_curve(-risk, risk, coverages)
    aurc["Oracle"] = compute_aurc(cov, risks)

    with open(os.path.join(args.out_dir, "token_aurc.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["selector", "aurc"])
        for k, v in aurc.items():
            w.writerow([k, f"{v:.6f}"])
    print("[dit-tok] AURC:", "  ".join(f"{k}={v:.5f}" for k, v in aurc.items()))

    # accept maps figure
    if example_maps:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            m = len(example_maps)
            fig, ax = plt.subplots(m, 4, figsize=(4 * 2.4, m * 2.4))
            if m == 1:
                ax = ax[None, :]
            cols = ["eps agreement s(p)", "x0 disagreement d(p)", "wavelet HF", "accept map"]
            for i, em in enumerate(example_maps):
                for j, key in enumerate(["s_eps", "d_x0", "wav", "accept"]):
                    a = ax[i, j]
                    a.imshow(em[key], cmap="magma"); a.set_xticks([]); a.set_yticks([])
                    if i == 0:
                        a.set_title(cols[j], fontsize=9)
                    if j == 0:
                        a.set_ylabel(f"t={em['t']}", fontsize=8)
            plt.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "accept_maps.png"), dpi=150,
                        bbox_inches="tight")
            print(f"[dit-tok] wrote {os.path.join(args.out_dir, 'accept_maps.png')}")
        except Exception as e:
            print(f"[dit-tok] map figure skipped: {e}")

    summary = dict(
        target_model=args.target_model, draft_model=args.draft_model,
        target_params_M=round(count_params(target) / 1e6, 2),
        draft_params_M=round(count_params(draft) / 1e6, 2),
        dataset="CIFAR-10", num_tokens=int(target.num_tokens),
        num_tokens_analyzed=int(n),
        accept_mean=float((merged["s_eps"] > (1 - args.tol)).mean()),
        accept_std=float((merged["s_eps"] > (1 - args.tol)).std()),
        mean_agreement=float(merged["s_eps"].mean()),
        aurc=aurc)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dit-tok] wrote {os.path.join(args.out_dir, 'summary.json')}")
    print(f"[dit-tok] accept_mean={summary['accept_mean']:.3f} "
          f"mean_agreement={summary['mean_agreement']:.3f}")


def _norm(x):
    mn, mx = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - mn) / (mx - mn + 1e-8), 0, 1)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Ti")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="results/dit_token_poc_v0")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_batches", type=int, default=40)
    ap.add_argument("--n_timesteps", type=int, default=8)
    ap.add_argument("--t_min", type=int, default=50)
    ap.add_argument("--t_max", type=int, default=950)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--n_example_maps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
