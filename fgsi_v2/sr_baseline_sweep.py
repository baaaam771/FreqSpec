#!/usr/bin/env python
"""
sr_baseline_sweep.py — SR analog of baseline_sweep.py (AAAI Table-2 style).

Answers the reviewer reflex for super-resolution: "is FreqSpec better than just
reducing the upscaler's denoising steps to match the speed?" Produces the
operating-point family on the SAME images:

    target_s{50,40,30}  (global step-reduction reference curve)
    freqspec_{strict,mid,default}  (verifier-controlled operating points)

Reuses the task-agnostic verifier via inference.speculative_general
(fgsr_refine / baseline_refine). SR setup: region = whole field, condition =
low-res image, no known-region blending, wavelet-only saliency.

Per-image, per-method it saves hr/lr/out(/usage_map) and a CSV row with
wall-time + NFE + accept rate + inline PSNR/SSIM/HH-PSNR. Trajectory-divergence
(LPIPSt) and LPIPS are computed afterwards by analyze_sr.py from saved images
(target_s50 output is the per-image reference).

Example:
    export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
    python sr_baseline_sweep.py \
        --target_id stabilityai/stable-diffusion-x4-upscaler \
        --draft_ckpt /mnt/HDD_12TB/bam_ki/ckpt_sr/draft_sr_final.pt --use_ema_draft \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/div2k/valid \
        --out_root  /mnt/HDD_12TB/bam_ki/results/sr_sweep_div2k \
        --num_images 100 --lr_size 128 --scale 4 \
        --target_steps 50 40 30 \
        --blend_temperature 0.10 \
        --x0_thr_strict 0.02 --x0_thr_loose 0.07 \
        --x0_strict_center 0.45 --x0_strict_width 0.12 \
        --drift_k_switch_threshold 0.006 --save_usage_maps
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sr_target_wrapper import SRTargetWrapper
from models.draft import DraftEpsUNet
from models.wavelet import DWT2D
from training.scheduler import DDPMSchedule
from inference.speculative_general import fgsr_refine, baseline_refine
from utils.metrics import psnr, ssim, hh_band_psnr


SR_TOL_PRESETS = [
    ("freqspec_strict",  0.01, 0.10),
    ("freqspec_mid",     0.02, 0.15),
    ("freqspec_default", 0.03, 0.30),
]


def build_method_list(args):
    methods = []
    if not args.freqspec_only:
        methods += [{"name": f"target_s{s}", "type": "target", "num_steps": s,
                     "tol_low": None, "tol_high": None} for s in args.target_steps]
    for name, tl, th in SR_TOL_PRESETS:
        methods.append({"name": name, "type": "freqspec",
                        "num_steps": args.num_steps, "tol_low": tl, "tol_high": th})
    return methods


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    return transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(img).unsqueeze(0)


def load_image_native(path, scale):
    """Load HR at native resolution, center-cropped to a multiple of 8*scale.

    For standard SR benchmarks (Set5/14, BSD100, Urban100) images vary in size
    and are mostly below 512, so the fixed 512 center-crop in load_image would
    upscale-then-evaluate (invalid SR). Native mode keeps the original HR and
    only crops to make latent dims (HR/scale) divisible by 8 (UNet) and 4 (patch).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    m = 8 * scale  # HR multiple so that latent = HR/scale is divisible by 8
    cw, ch = (w // m) * m, (h // m) * m
    cw, ch = max(cw, m), max(ch, m)
    left, top = (w - cw) // 2, (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    t = transforms.ToTensor()(img)
    t = transforms.Normalize([0.5] * 3, [0.5] * 3)(t)
    return t.unsqueeze(0)


def save_rgb(t, path):
    t = (t.float().clamp(-1, 1) + 1) / 2
    Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(path)


def save_gray(t, path):
    Image.fromarray((t[0, 0].float().cpu().numpy() * 255).clip(0, 255).astype("uint8")).save(path)


def build_manifest(args):
    root = Path(args.data_root)
    all_imgs = []
    for e in ("jpg", "jpeg", "png", "webp"):
        all_imgs += list(root.rglob(f"*.{e}"))
    all_imgs = sorted(str(p) for p in all_imgs)
    rng = random.Random(args.seed)
    rng.shuffle(all_imgs)
    chosen = all_imgs[:args.num_images]
    return [{"idx": i, "image_path": p, "prompt": args.prompt,
             "seed": args.seed * 100000 + i} for i, p in enumerate(chosen)]


def timed_run(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _prepare(target, item, args, device):
    if args.native_res:
        hr = load_image_native(item["image_path"], args.scale).to(device)
        Hh, Ww = hr.shape[-2:]
        hl, wl = Hh // args.scale, Ww // args.scale
    else:
        hr_size = args.lr_size * args.scale
        hr = load_image(item["image_path"], hr_size).to(device)
        hl = wl = args.lr_size
    lr = F.interpolate(hr, size=(hl, wl), mode="bicubic", align_corners=False).clamp(-1, 1)
    cond_lr, nl = target.prepare_lr_cond(lr, noise_level=args.noise_level)
    region = torch.ones(1, 1, hl, wl, device=device)
    torch.manual_seed(item["seed"])
    z_init = torch.randn(1, target.latent_ch, hl, wl, device=device)
    if target.available:
        cond_emb, uncond_emb, _ = target.get_text_embeddings(
            item["prompt"], batch_size=1, guidance_scale=args.guidance_scale)
    else:
        cond_emb, uncond_emb = None, None
    return hr, lr, cond_lr, nl, region, z_init, cond_emb, uncond_emb


def run_one(target, draft, sch, dwt, item, method, args, device):
    hr, lr, cond_lr, nl, region, z_init, cond_emb, uncond_emb = _prepare(
        target, item, args, device)
    extra = {"noise_level": nl}
    m_out = os.path.join(args.out_root, method["name"], f"img_{item['idx']:04d}")
    os.makedirs(m_out, exist_ok=True)

    with torch.no_grad():
        if method["type"] == "target":
            (z_out, stats), t_run = timed_run(lambda: baseline_refine(
                target, z_init.clone(), cond_lr, region, sch,
                num_inference_steps=method["num_steps"],
                guidance_scale=args.guidance_scale,
                cond_emb=cond_emb, uncond_emb=uncond_emb,
                known_z=None, blend_known=False, target_extra=extra), device)
            tgt_nfe, drf_nfe, acc = stats["target_calls"], 0, ""
            usage = None
        else:
            (z_out, stats), t_run = timed_run(lambda: fgsr_refine(
                target, draft, z_init.clone(), cond_lr, region, sch,
                num_inference_steps=method["num_steps"], K=args.K, patch_size=args.patch,
                t_spec_start_norm=args.t_spec_start, beta=args.beta,
                tol_low=method["tol_low"], tol_high=method["tol_high"],
                boundary_weight=0.0, mask_interior_weight=0.0,  # SR: wavelet-only
                guidance_scale=args.guidance_scale,
                cond_emb=cond_emb, uncond_emb=uncond_emb,
                known_z=None, blend_known=False,
                blend_temperature=args.blend_temperature,
                x0_thr_strict=args.x0_thr_strict, x0_thr_loose=args.x0_thr_loose,
                x0_strict_center=args.x0_strict_center, x0_strict_width=args.x0_strict_width,
                saliency_x0_coupling=args.saliency_x0_coupling,
                drift_k_switch_threshold=args.drift_k_switch_threshold,
                k_switch_threshold=args.k_switch_threshold,
                dwt=dwt, target_extra=extra,
                return_usage_map=args.save_usage_maps), device)
            tgt_nfe, drf_nfe = stats["target_calls"], stats["draft_calls"]
            acc = f"{stats['accept_rate']:.4f}"
            usage = stats.get("usage_map")

    out = target.decode_latent(z_out)
    save_rgb(hr, os.path.join(m_out, "hr.png"))
    save_rgb(out, os.path.join(m_out, "out.png"))
    if usage is not None:
        # upsample latent-res usage map to HR for inspection
        um = F.interpolate(usage, size=out.shape[-2:], mode="nearest")
        save_gray(um, os.path.join(m_out, "usage_map.png"))

    if target.available:
        with torch.no_grad():
            p = psnr(out, hr).item(); s = ssim(out, hr).item()
            hp = hh_band_psnr(out, hr, dwt).item()
    else:
        p = s = hp = float("nan")

    return [item["idx"], item["image_path"], method["name"], method["num_steps"],
            f"{t_run:.4f}", tgt_nfe, drf_nfe, acc, f"{p:.4f}", f"{s:.5f}", f"{hp:.4f}"]


def main(args):
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.target_dtype]
    target = SRTargetWrapper(model_id=args.target_id, device=device, dtype=dtype,
                             default_noise_level=args.noise_level)
    if not target.available:
        print("[sr-sweep] WARNING: target dummy mode (metrics meaningless)")

    if target.available:
        sch = DDPMSchedule(
            num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
            beta_start=target.scheduler_ref.config.beta_start,
            beta_end=target.scheduler_ref.config.beta_end,
            beta_schedule=getattr(target.scheduler_ref.config, "beta_schedule",
                                  "scaled_linear"),
            device=device)
    else:
        sch = DDPMSchedule(device=device)

    draft_kwargs = {"latent_ch": target.latent_ch,
                    "num_train_timesteps": sch.num_train_timesteps,
                    "cond_ch": 3, "use_mask": False}
    if args.draft_ckpt and os.path.isfile(args.draft_ckpt):
        ck = torch.load(args.draft_ckpt, map_location=device)
        sa = ck.get("args", {})
        if "draft_base_ch" in sa:
            draft_kwargs["base_ch"] = sa["draft_base_ch"]
            draft_kwargs["ch_mult"] = tuple(sa["draft_ch_mult"])
            draft_kwargs["t_dim"] = sa["draft_t_dim"]
        draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
        key = "ema_draft" if (args.use_ema_draft and ck.get("ema_draft")) else "draft"
        draft.load_state_dict(ck[key])
        print(f"[sr-sweep] loaded {key} from {args.draft_ckpt}")
    else:
        draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
        print("[sr-sweep] no draft ckpt -> random weights (dummy/smoke only)")

    dwt = DWT2D("haar").to(device)
    methods = build_method_list(args)
    manifest = build_manifest(args)

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sr-sweep] {len(manifest)} images x {len(methods)} methods: "
          f"{[m['name'] for m in methods]}")

    for method in methods:
        csv_path = os.path.join(args.out_root, method["name"], "results.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        done = set()
        if args.resume and os.path.isfile(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    done.add(int(row["idx"]))
        write_header = not (args.resume and os.path.isfile(csv_path))
        fcsv = open(csv_path, "a", newline="")
        w = csv.writer(fcsv)
        if write_header:
            w.writerow(["idx", "image_path", "method", "num_steps", "time_sec",
                        "target_nfe", "draft_nfe", "accept_rate",
                        "psnr", "ssim", "hh_psnr"])
        print(f"\n[sr-sweep] === {method['name']} ===")
        for item in manifest:
            if item["idx"] in done:
                continue
            row = run_one(target, draft, sch, dwt, item, method, args, device)
            w.writerow(row); fcsv.flush()
            print(f"  [{method['name']}] img {item['idx']:04d} t={row[4]}s "
                  f"tgt_nfe={row[5]} acc={row[7]} psnr={row[8]} hh={row[10]}")
        fcsv.close()
    print(f"\n[sr-sweep] done -> {args.out_root}")
    print("[sr-sweep] next: python analyze_sr.py --out_root <this> "
          "(LPIPS / LPIPSt vs target_s50)")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--draft_ckpt", type=str, default="")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-x4-upscaler")
    p.add_argument("--target_dtype", type=str, default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--use_ema_draft", action="store_true")
    p.add_argument("--num_images", type=int, default=100)
    p.add_argument("--lr_size", type=int, default=128)
    p.add_argument("--native_res", action="store_true",
                   help="evaluate at each image's native HR (crop to mult of 8*scale); "
                        "use for SR benchmarks where images vary in size / are <512")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--noise_level", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--target_steps", type=int, nargs="+", default=[50, 40, 30])
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--t_spec_start", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--blend_temperature", type=float, default=0.10)
    p.add_argument("--x0_thr_strict", type=float, default=0.02)
    p.add_argument("--x0_thr_loose", type=float, default=0.07)
    p.add_argument("--x0_strict_center", type=float, default=0.45)
    p.add_argument("--x0_strict_width", type=float, default=0.12)
    p.add_argument("--drift_k_switch_threshold", type=float, default=0.006)
    p.add_argument("--saliency_x0_coupling", type=float, default=0.0,
                   help="couple wavelet saliency into the x0 gate (SR freq ablation; 0=off)")
    p.add_argument("--freqspec_only", action="store_true",
                   help="skip target_sN baselines (for coupling reruns)")
    p.add_argument("--k_switch_threshold", type=float, default=0.60)
    p.add_argument("--save_usage_maps", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())