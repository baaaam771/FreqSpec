#!/usr/bin/env python
"""
sr_qualitative_compare.py — Target-30 vs Target-50 vs FreqSpec (oversmoothing check).

Target-30 (reduced-step) often *wins on PSNR* in SR while losing high-frequency
texture. This figure makes that visible and quantifies it: for each image we show
LR, Target-50 (reference), Target-30 (strong reduced-step baseline), and
FreqSpec-Aggressive (coupling 0.9, default tolerance), plus zoom crops on the
highest-frequency region. Each output is annotated with PSNR (vs HR) and a
high-frequency energy ratio HF% (mean gradient magnitude relative to Target-50);
a method that oversmooths shows high PSNR but low HF%.

The intended takeaway for the paper's §2 defense: Target-30's global PSNR edge can
coincide with reduced high-frequency energy (oversmoothing), so matched-cost PSNR
is not the whole story.

Run on the server (real weights). Example:
    python sr_qualitative_compare.py \
        --draft_ckpt /mnt/HDD_12TB/bam_ki/ckpt_sr/draft_sr_final.pt --use_ema_draft \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/sr_bench/Urban100_HR \
        --images img_004.png img_011.png img_024.png img_092.png \
        --out /mnt/HDD_12TB/bam_ki/results/sr_qual/oversmoothing.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sr_qualitative import (load_native, to_disp, hf_map, build_models, find_path)
from inference.speculative_general import fgsr_refine, baseline_refine
from utils.metrics import psnr as psnr_fn


def hf_energy(rgb01):
    """Mean gradient magnitude = scalar high-frequency energy."""
    return float(hf_map(rgb01).mean())


def main(args):
    device = torch.device(args.device)
    target, draft, sch, dwt = build_models(args, device)

    rows = []
    for name in args.images:
        pat = os.path.join(args.data_root, "**", name) if args.data_root else name
        path = find_path(pat)
        if path is None:
            print(f"[cmp] not found: {pat}"); continue
        hr = load_native(path, args.scale).to(device)
        Hh, Ww = hr.shape[-2:]
        hl, wl = Hh // args.scale, Ww // args.scale
        lr = F.interpolate(hr, size=(hl, wl), mode="bicubic", align_corners=False).clamp(-1, 1)
        lr_disp = F.interpolate(lr, size=(Hh, Ww), mode="bicubic", align_corners=False).clamp(-1, 1)
        cond_lr, nl = target.prepare_lr_cond(lr, noise_level=args.noise_level)
        region = torch.ones(1, 1, hl, wl, device=device)
        torch.manual_seed(args.seed)
        z0 = torch.randn(1, target.latent_ch, hl, wl, device=device)
        extra = {"noise_level": nl}

        with torch.no_grad():
            z50, _ = baseline_refine(target, z0.clone(), cond_lr, region, sch,
                                     num_inference_steps=50, exact_schedule=True,
                                     target_extra=extra)
            z30, _ = baseline_refine(target, z0.clone(), cond_lr, region, sch,
                                     num_inference_steps=30, exact_schedule=True,
                                     target_extra=extra)
            zfs, st = fgsr_refine(
                target, draft, z0.clone(), cond_lr, region, sch,
                num_inference_steps=50, exact_schedule=True, K=args.K, patch_size=args.patch,
                boundary_weight=0.0, mask_interior_weight=0.0,
                blend_temperature=0.10, x0_thr_strict=0.02, x0_thr_loose=0.07,
                x0_strict_center=0.45, x0_strict_width=0.12,
                saliency_x0_coupling=args.coupling,
                drift_k_switch_threshold=0.006, k_switch_threshold=0.60,
                tol_low=0.03, tol_high=0.30, dwt=dwt,
                known_z=None, blend_known=False, target_extra=extra)
            o50 = to_disp(target.decode_latent(z50))
            o30 = to_disp(target.decode_latent(z30))
            ofs = to_disp(target.decode_latent(zfs))
            hr_disp = to_disp(hr)
            p30 = psnr_fn(target.decode_latent(z30), hr).item()
            p50 = psnr_fn(target.decode_latent(z50), hr).item()
            pfs = psnr_fn(target.decode_latent(zfs), hr).item()

        e50 = hf_energy(o50)
        e30, efs = hf_energy(o30), hf_energy(ofs)
        # zoom on highest-HF region (from Target-50)
        hf = hf_map(o50); zs = args.zoom
        if Hh > zs and Ww > zs:
            dens = uniform_filter(hf, size=zs)
            yc, xc = np.unravel_index(np.argmax(dens), dens.shape)
            y0 = int(np.clip(yc - zs // 2, 0, Hh - zs)); x0 = int(np.clip(xc - zs // 2, 0, Ww - zs))
        else:
            y0 = x0 = 0; zs = min(Hh, Ww)
        rows.append(dict(
            name=name, lr=to_disp(lr_disp), o50=o50, o30=o30, ofs=ofs,
            z30=o30[y0:y0+zs, x0:x0+zs], zfs=ofs[y0:y0+zs, x0:x0+zs],
            p50=p50, p30=p30, pfs=pfs,
            hf50=100.0, hf30=100.0*e30/e50, hffs=100.0*efs/e50,
            accept=st["accept_rate"]))
        print(f"[cmp] {name}: PSNR t50={p50:.2f} t30={p30:.2f} fs={pfs:.2f} | "
              f"HF% t30={100*e30/e50:.1f} fs={100*efs/e50:.1f} | accept={st['accept_rate']:.3f}")

    if not rows:
        print("[cmp] nothing to plot"); return

    titles = ["LR (bicubic x4)", "Target-50", "Target-30",
              f"FreqSpec (cpl {args.coupling})", "Target-30 (zoom)", "FreqSpec (zoom)"]
    ncol = 6
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 2.6, len(rows) * 2.7))
    if len(rows) == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        cells = [(r["lr"], None),
                 (r["o50"], f"PSNR {r['p50']:.2f}\nHF {r['hf50']:.0f}%"),
                 (r["o30"], f"PSNR {r['p30']:.2f}\nHF {r['hf30']:.0f}%"),
                 (r["ofs"], f"PSNR {r['pfs']:.2f}\nHF {r['hffs']:.0f}%\nacc {r['accept']:.2f}"),
                 (r["z30"], "Target-30"),
                 (r["zfs"], "FreqSpec")]
        for j, (img, cap) in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if cap:
                ax.text(0.03, 0.04, cap, transform=ax.transAxes, color="white",
                        fontsize=7, va="bottom", ha="left",
                        bbox=dict(facecolor="black", alpha=0.45, pad=1.5, lw=0))
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(r["name"], fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[cmp] saved {args.out} and {pdf}")

    mh30 = np.mean([r["hf30"] for r in rows]); mhfs = np.mean([r["hffs"] for r in rows])
    mp30 = np.mean([r["p30"] for r in rows]); mpfs = np.mean([r["pfs"] for r in rows])
    print(f"[cmp] mean PSNR  Target-30={mp30:.2f}  FreqSpec={mpfs:.2f}")
    print(f"[cmp] mean HF%   Target-30={mh30:.1f}  FreqSpec={mhfs:.1f}  "
          f"(<100 = oversmoothed vs Target-50)")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--images", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, default="./sr_qual/oversmoothing.png")
    p.add_argument("--draft_ckpt", type=str, default="")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-x4-upscaler")
    p.add_argument("--target_dtype", type=str, default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--use_ema_draft", action="store_true")
    p.add_argument("--coupling", type=float, default=0.9)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--noise_level", type=int, default=20)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--zoom", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--num_images", type=int, default=4)  # unused; kept for build_models
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
