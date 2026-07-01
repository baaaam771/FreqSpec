#!/usr/bin/env python
"""
dit_token_fid.py -- Phase 4: FID-10k for DiT token-mixing on ImageNet-64.

For each method (target / draft / freqspec / random) it generates N samples with
the exact same K=1 token-mixing sampler used elsewhere, writes them to PNG, and
computes FID against a reference image folder (ImageNet-64 validation). Because
pixel-std cannot separate methods when the draft does not collapse, FID is the
decisive metric here: FreqSpec-token should score below Random-token at the same
target-token budget.

Requires clean-fid:  pip install clean-fid

Usage:
    python dit_token_fid.py \
        --target /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt --target_model DiT-S \
        --draft  /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --ref_dir /mnt/HDD_12TB/bam_ki/datasets/imagenet64/val \
        --n_samples 10000 --steps 50 --accept_ratio 0.5 \
        --out_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/fid_ar0.5
"""
import argparse
import json
import os

import torch
from torchvision.utils import save_image

from models.dit import build_dit, count_params
from training.scheduler import DDPMSchedule
# reuse the exact sampler internals so FID matches the analyzed sampler
from dit_token_sampler import load_dit, sample


@torch.no_grad()
def gen_method(method, target, draft, sch, ts, args, dev, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    idx = 0
    accs = []
    while n_done < args.n_samples:
        b = min(args.batch, args.n_samples - n_done)
        y = torch.randint(0, args.num_classes, (b,), device=dev)
        z = torch.randn(b, 3, args.img_size, args.img_size, device=dev)
        x, accept = sample(method, target, draft, sch, ts, y, z, args, dev)
        accs.append(float(accept))
        x = ((x.clamp(-1, 1) + 1) / 2)
        for j in range(b):
            save_image(x[j], os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1
        n_done += b
        if n_done % (args.batch * 10) == 0 or n_done == args.n_samples:
            print(f"[fid] {method}: {n_done}/{args.n_samples}")
    return sum(accs) / max(len(accs), 1)


def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    print(f"[fid] target {args.target_model} {count_params(target)/1e6:.1f}M | "
          f"draft {args.draft_model} {count_params(draft)/1e6:.1f}M | "
          f"accept_ratio={args.accept_ratio}")

    from cleanfid import fid as cfid

    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    methods = args.methods.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for m in methods:
        gdir = os.path.join(args.out_dir, f"samples_{m}")
        acc = gen_method(m, target, draft, sch, ts, args, dev, gdir)
        score = cfid.compute_fid(gdir, args.ref_dir, mode="clean",
                                 num_workers=args.workers)
        tgt_use = (1.0 - acc) if m in ("freqspec", "random") else \
                  (0.0 if m == "draft" else 1.0)
        results[m] = dict(fid=round(float(score), 3),
                          accept=round(acc, 4),
                          target_token_usage=round(tgt_use, 4))
        print(f"[fid] {m:9s} FID={score:.3f}  accept={acc:.3f}  tgt_use={tgt_use:.3f}")

    out = dict(accept_ratio=args.accept_ratio, n_samples=args.n_samples,
               steps=args.steps, ref_dir=args.ref_dir, methods=results)
    with open(os.path.join(args.out_dir, "fid_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[fid] wrote {os.path.join(args.out_dir, 'fid_summary.json')}")

    if "freqspec" in results and "random" in results:
        gap = results["random"]["fid"] - results["freqspec"]["fid"]
        print(f"[fid] FreqSpec vs Random: FID {results['freqspec']['fid']} vs "
              f"{results['random']['fid']}  (FreqSpec better by {gap:+.3f})")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--ref_dir", type=str, required=True,
                    help="reference real images (e.g. ImageNet-64 val folder)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--accept_ratio", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--methods", type=str, default="target,draft,freqspec,random")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
