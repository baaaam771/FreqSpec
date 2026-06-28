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


class _ImageFolderDS(torch.utils.data.Dataset):
    """Top-level (picklable) random-crop image dataset; single dummy class 0."""
    def __init__(self, paths, img_size, train=False):
        from torchvision import transforms
        self.paths = paths
        ops = [transforms.RandomCrop(img_size, pad_if_needed=True)]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB")), 0


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
    if args.dataset == "imagenet":
        tf = transforms.Compose([transforms.Resize(args.img_size),
                                 transforms.CenterCrop(args.img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])
        ds = datasets.ImageFolder(args.data_root, transform=tf)
    elif args.dataset == "imagefolder":
        import glob
        paths = []
        for e in ("png", "jpg", "jpeg"):
            paths += glob.glob(os.path.join(args.data_root, "**", f"*.{e}"), recursive=True)
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"no images under {args.data_root}")
        ds = _ImageFolderDS(paths, args.img_size, train=False)
    else:
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

    chunks = {k: [] for k in ("d_x0", "s_eps", "scos", "tnorm", "wav")}
    t_ids = []
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
            # direction-only agreement: per-token mean cosine between draft/target eps
            dot = F.avg_pool2d((eps_d * eps_t).sum(1, keepdim=True), p, stride=p)
            nd = F.avg_pool2d(eps_d.pow(2).sum(1, keepdim=True), p, stride=p).sqrt()
            nt = F.avg_pool2d(eps_t.pow(2).sum(1, keepdim=True), p, stride=p).sqrt()
            scos = dot / (nd * nt + 1e-8)                        # [-1,1], higher=agree
            tnorm = token_pool(x0_t, p)                          # target content magnitude
            wav = lwd_wavelet_saliency(x_t, dwt, target_size=(x_t.shape[-2] // p,
                                                              x_t.shape[-1] // p))
            for k, v in (("d_x0", d_x0), ("s_eps", s_eps), ("scos", scos),
                         ("tnorm", tnorm), ("wav", wav)):
                chunks[k].append(v.flatten().cpu().numpy())
            t_ids.append(np.full(d_x0.numel(), int(tv), dtype=np.int32))
            if len(example_maps) < args.n_example_maps and int(tv) == int(ts_grid[len(ts_grid) // 2]):
                sm = s_eps[0, 0].cpu().numpy()
                thr = np.quantile(sm, 1.0 - args.accept_ratio)
                example_maps.append(dict(
                    t=int(tv),
                    s_eps=sm,
                    d_x0=d_x0[0, 0].cpu().numpy(),
                    wav=wav[0, 0].cpu().numpy(),
                    accept=(sm >= thr).astype(np.float32)))
        nb += 1
        if nb % 5 == 0:
            print(f"[dit-tok] {nb}/{args.num_batches} batches")

    merged = {k: np.concatenate(v) for k, v in chunks.items()}
    t_id = np.concatenate(t_ids)
    n = merged["d_x0"].size
    print(f"[dit-tok] {n} tokens total")

    risk = merged["d_x0"]
    rng = np.random.default_rng(0)
    selectors = {
        "Random":            rng.random(n),
        "Eps agreement":     merged["s_eps"],
        "Eps-cosine":        merged["scos"],
        "Token-norm":        1.0 - _norm(merged["tnorm"]),
        "Frequency-token":   1.0 - _norm(merged["wav"]),
    }
    coverages = np.linspace(0.05, 1.0, 40)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- primary metric: per-timestep AURC, risk min-max normalized within each
    # timestep so noise levels are comparable, then averaged over timesteps.
    # (Pooling all timesteps is misleading: x0-risk explodes at high t while
    # eps-disagreement shrinks, so a single pool lets high-t tokens dominate the
    # ranking even though within any timestep eps-agreement predicts risk
    # near-perfectly. The sampler decides per step, so per-timestep is the
    # operationally correct evaluation.)
    uniq_t = np.unique(t_id)

    def per_t_aurc(conf):
        vals = []
        for tv in uniq_t:
            m = t_id == tv
            r = risk[m]
            rs = r.max() - r.min()
            rn = (r - r.min()) / rs if rs > 1e-12 else np.zeros_like(r)
            cov, risks = risk_coverage_curve(conf[m], rn, coverages)
            vals.append(compute_aurc(cov, risks))
        return float(np.mean(vals)), vals

    aurc, aurc_per_t = {}, {}
    for name, conf in selectors.items():
        aurc[name], aurc_per_t[name] = per_t_aurc(conf)
    aurc["Oracle"], aurc_per_t["Oracle"] = per_t_aurc(-risk)

    # pooled (for reference / to document the artifact)
    pooled = {}
    for name, conf in selectors.items():
        cov, risks = risk_coverage_curve(conf, risk, coverages)
        pooled[name] = compute_aurc(cov, risks)
    cov, risks = risk_coverage_curve(-risk, risk, coverages)
    pooled["Oracle"] = compute_aurc(cov, risks)

    with open(os.path.join(args.out_dir, "token_aurc.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["selector", "aurc_per_timestep", "aurc_pooled"])
        for k in aurc:
            w.writerow([k, f"{aurc[k]:.6f}", f"{pooled[k]:.6f}"])
    print("[dit-tok] per-timestep AURC (primary):",
          "  ".join(f"{k}={v:.5f}" for k, v in aurc.items()))
    print("[dit-tok] pooled AURC (reference) :",
          "  ".join(f"{k}={v:.5f}" for k, v in pooled.items()))

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

    # accept at the chosen operating point: per-timestep, accept the top
    # accept_ratio fraction of tokens by eps-agreement (mirrors the SR tolerance
    # knob; gives a controllable accept ratio and visible reject structure).
    accept_mask = np.zeros(n, dtype=bool)
    for tv in uniq_t:
        m = t_id == tv
        se = merged["s_eps"][m]
        thr = np.quantile(se, 1.0 - args.accept_ratio)
        accept_mask[m] = se >= thr
    summary = dict(
        target_model=args.target_model, draft_model=args.draft_model,
        target_params_M=round(count_params(target) / 1e6, 2),
        draft_params_M=round(count_params(draft) / 1e6, 2),
        dataset="CIFAR-10", num_tokens=int(target.num_tokens),
        num_tokens_analyzed=int(n),
        accept_ratio_setpoint=args.accept_ratio,
        accept_mean=float(accept_mask.mean()),
        mean_agreement=float(merged["s_eps"].mean()),
        aurc_per_timestep=aurc, aurc_pooled=pooled)
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
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "imagefolder", "imagenet"])
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
    ap.add_argument("--accept_ratio", type=float, default=0.7,
                    help="operating point: per-timestep fraction of tokens accepted "
                         "by eps-agreement (controls accept map + accept_mean)")
    ap.add_argument("--n_example_maps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
