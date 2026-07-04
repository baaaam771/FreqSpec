#!/usr/bin/env python
"""
dit_heterogeneity.py — quantify the spatial heterogeneity of the target's
temporal eps change, the quantity that determines whether token-wise step
allocation can beat uniform step reduction.

A7 finding to substantiate: under target-eps reuse, random ~= oracle ~= anchor
selection and the hard-ratio sweep is flat — i.e. on ImageNet-64 DiT-S the
per-token eps change between nearby steps is nearly UNIFORM across tokens, so
refreshing any subset is as good as any other and uniform step reduction is
already the optimal allocation. This script measures that directly.

Per denoising step t it computes the per-token temporal change
    delta_i(t) = || eps_i(z_t, t) - eps_i(z_{t'}, t') ||^2   (consecutive steps)
and reports, per step and pooled:
    CV        = std_i / mean_i  (across tokens; the heterogeneity index)
    p90/p50   = tail ratio (how much harder the hardest tokens are)
    top-r share = fraction of total change carried by the top-r tokens
                  (r = 0.3: if ~0.3, change is uniform; >>0.3, concentrated)
    spatial autocorr of delta (Moran-like neighbor correlation on the grid)

Interpretation: token-wise allocation can only beat uniform step reduction
when CV / tail ratio are large (concentrated change). Low values on IN-64
DiT-S explain the A7 null result and predict where the method pays off
(inpainting masks, high-res latent grids).

Usage:
    python dit_heterogeneity.py \
        --target ckpt/target.pt --target_model DiT-S \
        --img_size 64 --patch 4 --num_classes 1000 \
        --n_traj 64 --steps 50 --out results/dit_in64_sparse/heterogeneity
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    # load_dit only needs target-style args
    args.draft = args.target
    target = load_dit(args.target, args.target_model, args, dev)
    p = args.patch
    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    g = torch.Generator(device=dev).manual_seed(args.seed)

    per_step = {k: [[] for _ in ts] for k in ("cv", "p90_p50", "topr_share",
                                              "neigh_corr")}
    n_done = 0
    while n_done < args.n_traj:
        b = min(args.batch, args.n_traj - n_done)
        y = torch.randint(0, args.num_classes, (b,), device=dev, generator=g)
        z = torch.randn(b, 3, args.img_size, args.img_size, device=dev,
                        generator=g)
        prev_eps = None
        for i, t in enumerate(ts):
            t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
            tt = torch.full((b,), int(t), device=dev, dtype=torch.long)
            eps = target(z, tt, y)
            if prev_eps is not None:
                d2 = F.avg_pool2d((eps - prev_eps).pow(2).mean(1, keepdim=True),
                                  p, stride=p)                       # [B,1,h,w]
                d = d2.flatten(1)                                    # [B,N]
                mean = d.mean(1); std = d.std(1)
                per_step["cv"][i].extend((std / (mean + 1e-12)).tolist())
                q = torch.quantile(d, torch.tensor([0.5, 0.9], device=dev),
                                   dim=1)
                per_step["p90_p50"][i].extend(
                    (q[1] / (q[0] + 1e-12)).tolist())
                N = d.shape[1]
                k = max(1, int(round(args.topr * N)))
                top = d.topk(k, dim=1).values.sum(1)
                per_step["topr_share"][i].extend(
                    (top / (d.sum(1) + 1e-12)).tolist())
                # neighbor correlation on the token grid (spatial structure)
                dm = d2 - d2.mean(dim=(2, 3), keepdim=True)
                right = (dm[..., :, :-1] * dm[..., :, 1:]).mean(dim=(1, 2, 3))
                down = (dm[..., :-1, :] * dm[..., 1:, :]).mean(dim=(1, 2, 3))
                var = dm.pow(2).mean(dim=(1, 2, 3))
                per_step["neigh_corr"][i].extend(
                    (0.5 * (right + down) / (var + 1e-12)).tolist())
            prev_eps = eps
            z, _ = sch.ddim_step(z, eps, int(t), t_prev, eta=0.0)
        n_done += b
        print(f"[hetero] {n_done}/{args.n_traj}")

    def agg(key):
        vals = [np.mean(v) for v in per_step[key] if v]
        return dict(per_step=[round(float(x), 4) for x in vals],
                    pooled=round(float(np.mean(vals)), 4),
                    max=round(float(np.max(vals)), 4))

    out = dict(model=args.target_model, img_size=args.img_size,
               steps=args.steps, n_traj=args.n_traj, topr=args.topr,
               metrics={k: agg(k) for k in per_step})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "heterogeneity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[hetero] pooled CV={out['metrics']['cv']['pooled']} "
          f"p90/p50={out['metrics']['p90_p50']['pooled']} "
          f"top-{args.topr:g} share={out['metrics']['topr_share']['pooled']} "
          f"(uniform baseline={args.topr:g}) "
          f"neigh corr={out['metrics']['neigh_corr']['pooled']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        steps_ax = list(range(1, args.steps))
        for ax, key, lbl, base in (
                (axes[0], "cv", "across-token CV of temporal eps change", None),
                (axes[1], "topr_share",
                 f"top-{args.topr:g} token share of total change", args.topr),
                (axes[2], "neigh_corr", "neighbor correlation", 0.0)):
            vals = [np.mean(v) for v in per_step[key] if v]
            ax.plot(steps_ax[:len(vals)], vals, lw=2)
            if base is not None:
                ax.axhline(base, ls="--", c="grey", lw=1,
                           label="uniform baseline")
                ax.legend(fontsize=7)
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("denoising step")
        fig.tight_layout()
        fp = os.path.join(args.out, "heterogeneity.png")
        fig.savefig(fp, dpi=200)
        print(f"[hetero] wrote {fp}")
    except Exception as e:  # plotting is optional
        print(f"[hetero] plot skipped: {e}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--n_traj", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--topr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
