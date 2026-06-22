#!/usr/bin/env python
"""
sr_qualitative.py — qualitative figure for FreqSpec-SR.

For each image, renders a row of panels that makes the core claim visible:
the draft is used on smooth / low-frequency regions while the frozen target
verifies high-frequency texture and edges.

    LR (bicubic x4) | Target-50 | FreqSpec | Draft-usage w(p) | High-freq map | Zoom

The draft-usage map (brighter = more draft) should be visually ANTI-correlated
with the high-frequency map. We quantify this per image with the Pearson
correlation r between draft usage and low-frequency content (1 - HF); a positive
r means the draft concentrates where the image is locally smooth. The accept
rate and r are annotated on the usage panel.

Run on the server (real weights). Example:
    python sr_qualitative.py \
        --draft_ckpt /mnt/HDD_12TB/bam_ki/ckpt_sr/draft_sr_final.pt --use_ema_draft \
        --images img_073.png img_092.png img_004.png \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/sr_bench/Urban100_HR \
        --out /mnt/HDD_12TB/bam_ki/results/sr_qual/urban_usage.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sr_target_wrapper import SRTargetWrapper
from models.draft import DraftEpsUNet
from models.wavelet import DWT2D
from training.scheduler import DDPMSchedule
from inference.speculative_general import fgsr_refine, baseline_refine


def load_native(path, scale):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    m = 8 * scale
    cw, ch = max((w // m) * m, m), max((h // m) * m, m)
    left, top = (w - cw) // 2, (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    t = transforms.ToTensor()(img)
    return transforms.Normalize([0.5] * 3, [0.5] * 3)(t).unsqueeze(0)


def to_disp(t):
    """[-1,1] CHW tensor -> HWC uint8-ish float [0,1]."""
    return ((t[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def luminance(rgb01):
    return rgb01 @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def hf_map(rgb01):
    """Gradient-magnitude high-frequency map, normalized to [0,1]."""
    y = luminance(rgb01)
    gx = np.abs(np.gradient(y, axis=1))
    gy = np.abs(np.gradient(y, axis=0))
    g = np.sqrt(gx ** 2 + gy ** 2)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


def build_models(args, device):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.target_dtype]
    target = SRTargetWrapper(model_id=args.target_id, device=device, dtype=dtype,
                             default_noise_level=args.noise_level)
    if target.available:
        sch = DDPMSchedule(
            num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
            beta_start=target.scheduler_ref.config.beta_start,
            beta_end=target.scheduler_ref.config.beta_end,
            beta_schedule=getattr(target.scheduler_ref.config, "beta_schedule",
                                  "scaled_linear"), device=device)
    else:
        sch = DDPMSchedule(device=device)
    dk = {"latent_ch": target.latent_ch, "num_train_timesteps": sch.num_train_timesteps,
          "cond_ch": 3, "use_mask": False}
    if args.draft_ckpt and os.path.isfile(args.draft_ckpt):
        ck = torch.load(args.draft_ckpt, map_location=device)
        sa = ck.get("args", {})
        if "draft_base_ch" in sa:
            dk["base_ch"] = sa["draft_base_ch"]; dk["ch_mult"] = tuple(sa["draft_ch_mult"])
            dk["t_dim"] = sa["draft_t_dim"]
        draft = DraftEpsUNet(**dk).to(device).eval()
        key = "ema_draft" if (args.use_ema_draft and ck.get("ema_draft")) else "draft"
        draft.load_state_dict(ck[key]); print(f"[sr-qual] loaded {key}")
    else:
        draft = DraftEpsUNet(**dk).to(device).eval()
        print("[sr-qual] no ckpt -> random draft (dummy only)")
    return target, draft, sch, DWT2D("haar").to(device)


def resolve_paths(args):
    if args.images and args.data_root:
        return [os.path.join(args.data_root, "**", im) for im in args.images], args.images
    if args.images:
        return args.images, [os.path.basename(p) for p in args.images]
    paths = []
    for e in ("png", "jpg", "jpeg"):
        paths += glob.glob(os.path.join(args.data_root, "**", f"*.{e}"), recursive=True)
    paths = sorted(paths)[:args.num_images]
    return paths, [os.path.basename(p) for p in paths]


def find_path(pattern):
    if os.path.isfile(pattern):
        return pattern
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def main(args):
    device = torch.device(args.device)
    target, draft, sch, dwt = build_models(args, device)
    patterns, names = resolve_paths(args)

    rows = []
    for pat, name in zip(patterns, names):
        path = find_path(pat)
        if path is None:
            print(f"[sr-qual] not found: {pat}"); continue
        hr = load_native(path, args.scale).to(device)
        Hh, Ww = hr.shape[-2:]
        hl, wl = Hh // args.scale, Ww // args.scale
        lr = F.interpolate(hr, size=(hl, wl), mode="bicubic", align_corners=False).clamp(-1, 1)
        lr_disp = F.interpolate(lr, size=(Hh, Ww), mode="bicubic", align_corners=False).clamp(-1, 1)
        cond_lr, nl = target.prepare_lr_cond(lr, noise_level=args.noise_level)
        region = torch.ones(1, 1, hl, wl, device=device)
        torch.manual_seed(args.seed)
        z_init = torch.randn(1, target.latent_ch, hl, wl, device=device)
        extra = {"noise_level": nl}

        with torch.no_grad():
            z_tgt, _ = baseline_refine(target, z_init.clone(), cond_lr, region, sch,
                                       num_inference_steps=50, target_extra=extra)
            z_spec, st = fgsr_refine(
                target, draft, z_init.clone(), cond_lr, region, sch,
                num_inference_steps=50, K=args.K, patch_size=args.patch,
                boundary_weight=0.0, mask_interior_weight=0.0,
                blend_temperature=0.10, x0_thr_strict=0.02, x0_thr_loose=0.07,
                x0_strict_center=0.45, x0_strict_width=0.12,
                drift_k_switch_threshold=0.006, k_switch_threshold=0.60,
                tol_low=0.03, tol_high=0.30, dwt=dwt,
                known_z=None, blend_known=False,
                return_usage_map=True, target_extra=extra)
            out_tgt = to_disp(target.decode_latent(z_tgt))
            out_spec = to_disp(target.decode_latent(z_spec))

        usage = st["usage_map"]  # [1,1,hl,wl] in [0,1], higher=more draft
        usage_full = F.interpolate(usage, size=(Hh, Ww), mode="nearest")[0, 0].numpy()
        hf = hf_map(out_tgt)
        # correlation between draft usage and low-frequency content (1 - HF)
        u = usage_full.flatten(); lf = (1.0 - hf).flatten()
        r = float(np.corrcoef(u, lf)[0, 1]) if u.std() > 1e-6 else float("nan")

        # zoom on highest-HF region
        zs = args.zoom
        if Hh > zs and Ww > zs:
            from scipy.ndimage import uniform_filter
            dens = uniform_filter(hf, size=zs)
            yc, xc = np.unravel_index(np.argmax(dens), dens.shape)
            y0 = int(np.clip(yc - zs // 2, 0, Hh - zs))
            x0 = int(np.clip(xc - zs // 2, 0, Ww - zs))
        else:
            y0 = x0 = 0; zs = min(Hh, Ww)
        zoom = out_spec[y0:y0 + zs, x0:x0 + zs]

        rows.append(dict(name=name, lr=to_disp(lr_disp), tgt=out_tgt, spec=out_spec,
                         usage=usage_full, hf=hf, zoom=zoom,
                         accept=st["accept_rate"], nfe=st["target_calls"], r=r))
        print(f"[sr-qual] {name}: accept={st['accept_rate']:.3f} "
              f"tgt_nfe={st['target_calls']} corr(usage,1-HF)={r:+.3f}")

    if not rows:
        print("[sr-qual] nothing to plot"); return

    ncol = 6
    titles = ["LR (bicubic x4)", "Target-50", "FreqSpec", "Draft usage w(p)",
              "High-freq map", "FreqSpec (zoom)"]
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 2.6, len(rows) * 2.7))
    if len(rows) == 1:
        axes = axes[None, :]
    for i, row in enumerate(rows):
        panels = [row["lr"], row["tgt"], row["spec"], None, None, row["zoom"]]
        for j in range(ncol):
            ax = axes[i, j]
            if j == 3:
                im = ax.imshow(row["usage"], cmap="magma", vmin=0, vmax=1)
                ax.text(0.03, 0.06,
                        f"acc={row['accept']:.2f}\nNFE={row['nfe']:.0f}\nr={row['r']:+.2f}",
                        transform=ax.transAxes, color="white", fontsize=7,
                        va="bottom", ha="left",
                        bbox=dict(facecolor="black", alpha=0.4, pad=1.5, lw=0))
            elif j == 4:
                ax.imshow(row["hf"], cmap="viridis", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(panels[j], 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(row["name"], fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[sr-qual] saved {args.out} and {pdf}")

    mean_r = np.nanmean([row["r"] for row in rows])
    print(f"[sr-qual] mean corr(draft-usage, low-freq) = {mean_r:+.3f}  "
          f"(positive => draft concentrates on smooth regions)")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--images", type=str, nargs="*", default=None,
                   help="filenames (with --data_root) or full paths")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--out", type=str, default="./sr_qual/usage.png")
    p.add_argument("--draft_ckpt", type=str, default="")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-x4-upscaler")
    p.add_argument("--target_dtype", type=str, default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--use_ema_draft", action="store_true")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--noise_level", type=int, default=20)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--zoom", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
