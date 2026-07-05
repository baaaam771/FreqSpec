#!/usr/bin/env python
"""
dit_paired_compare.py — same-seed paired comparison of dense reduced-step
vs. r=0 anchored reuse, against a shared full-quality reference.

FID pools distributional variance; this isolates the per-sample question:
starting from the SAME initial noise and class label, which method's output
stays closer to the 50-step dense trajectory? For each sample we generate
    ref   : dense S=50            (same z0, y)
    A     : dense S=A_steps       (same z0, y)
    B     : r=0 reuse S=B_steps, cp  (same z0, y)
and report per-image LPIPS(x, ref), pixel MSE(x, ref), the paired win rate
P[LPIPS_B < LPIPS_A] with a normal-approximation sign-test p-value, and the
direct deviation MSE(A, B).

Usage:
    python dit_paired_compare.py \
        --target ckpt/target.pt --target_model DiT-S \
        --draft ckpt/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --pair 10,50,5 --pair 15,30,2 \
        --n_samples 256 --batch 64 \
        --out results/dit_in64_sparse/paired_compare
(--pair A_steps,B_steps,B_cp)
"""
import argparse
import json
import math
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit
from dit_sparse_sampler import sample_sparse


def cfg(base, kind, steps, cp=1):
    a = Namespace(**vars(base))
    a.steps = steps
    if kind == "dense":
        a.selector, a.hard_ratio, a.cache_period = "dense", 0.3, 5
    else:
        a.selector, a.easy_source = "random", "target_cache"
        a.hard_ratio, a.cache_period = 0.0, cp
    return a


@torch.no_grad()
def gen(target, draft, sch, a, y, z, dev):
    ts = sch.get_ddim_schedule_exact(a.steps).tolist()
    x, *_ = sample_sparse(target, draft, sch, ts, y, z.clone(), a, dev, None)
    return x.clamp(-1, 1)


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4,
                       beta_end=0.02, beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    base = Namespace(selector="dense", suffix_mode="cache_attn",
                     hard_ratio=0.0, split=0.0, cache_period=5,
                     refresh_every=0, dense_until=1.0,
                     easy_source="target_cache", patch=args.patch, steps=50)
    pairs = [tuple(int(v) for v in p.split(",")) for p in args.pair]
    os.makedirs(args.out, exist_ok=True)
    results = []

    for (a_steps, b_steps, b_cp) in pairs:
        g = torch.Generator(device=dev).manual_seed(args.seed)
        lp_a, lp_b, mse_a, mse_b, dev_ab = [], [], [], [], []
        n_done = 0
        while n_done < args.n_samples:
            b = min(args.batch, args.n_samples - n_done)
            y = torch.randint(0, args.num_classes, (b,), device=dev,
                              generator=g)
            z = torch.randn(b, 3, args.img_size, args.img_size,
                            device=dev, generator=g)
            ref = gen(target, draft, sch, cfg(base, "dense", args.ref_steps),
                      y, z, dev)
            xa = gen(target, draft, sch, cfg(base, "dense", a_steps),
                     y, z, dev)
            xb = gen(target, draft, sch, cfg(base, "r0", b_steps, b_cp),
                     y, z, dev)
            lp_a.extend(lp(xa, ref).flatten().tolist())
            lp_b.extend(lp(xb, ref).flatten().tolist())
            mse_a.extend((xa - ref).pow(2).mean(dim=(1, 2, 3)).tolist())
            mse_b.extend((xb - ref).pow(2).mean(dim=(1, 2, 3)).tolist())
            dev_ab.extend((xa - xb).pow(2).mean(dim=(1, 2, 3)).tolist())
            n_done += b
            print(f"[paired] s{a_steps} vs r0-s{b_steps}cp{b_cp}: "
                  f"{n_done}/{args.n_samples}")
        la, lb = np.array(lp_a), np.array(lp_b)
        nt = la != lb                                  # exclude exact ties
        wins = float((lb[nt] < la[nt]).mean()) if nt.any() else 0.5
        n = max(int(nt.sum()), 1)
        zst = (wins - 0.5) / math.sqrt(0.25 / n)      # sign test, normal approx
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(zst) / math.sqrt(2))))
        row = dict(
            dense_steps=a_steps, r0_steps=b_steps, r0_cp=b_cp,
            n=n, ref_steps=args.ref_steps,
            lpips_dense=dict(mean=round(float(la.mean()), 4),
                             std=round(float(la.std()), 4)),
            lpips_r0=dict(mean=round(float(lb.mean()), 4),
                          std=round(float(lb.std()), 4)),
            lpips_paired_delta=round(float((lb - la).mean()), 4),
            r0_win_rate=round(wins, 4), sign_test_p=round(pval, 5),
            mse_dense=round(float(np.mean(mse_a)), 6),
            mse_r0=round(float(np.mean(mse_b)), 6),
            mse_dense_vs_r0=round(float(np.mean(dev_ab)), 6))
        results.append(row)
        print(f"[paired] dense s{a_steps} LPIPS {row['lpips_dense']['mean']} "
              f"| r0 s{b_steps}cp{b_cp} LPIPS {row['lpips_r0']['mean']} "
              f"| r0 win rate {wins:.3f} (p={pval:.4g})")

    with open(os.path.join(args.out, "paired_compare.json"), "w") as f:
        json.dump(dict(seed=args.seed, results=results), f, indent=2)
    print(f"[paired] wrote {os.path.join(args.out, 'paired_compare.json')}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--pair", action="append", required=True,
                    help="dense_steps,r0_steps,r0_cp (repeatable)")
    ap.add_argument("--ref_steps", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
