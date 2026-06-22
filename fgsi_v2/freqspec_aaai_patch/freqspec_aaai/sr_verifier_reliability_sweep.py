#!/usr/bin/env python
"""
sr_verifier_reliability_sweep.py — SR analog of verifier_reliability_sweep.py.

Runs FreqSpec-SR with collect_patch_logs=True and dumps one .pt log per
image-seed containing the per-patch quantities needed for the risk-coverage /
AURC / FAR verifier-reliability analysis (AAAI Table-5 style):

    d_x0, s_eps, w, saliency, wav, t_norm   (+ image_id, seed)

The log format is IDENTICAL to the inpainting sweep, so the existing analyzer
runs unchanged on SR logs:

    python analyze_verifier_reliability.py \
        --logs_dir <out>/patch_logs --out_dir <out>/analysis

SR setup: region = whole field, condition = low-res image, wavelet-only
saliency, no known-region blending. Because the whole field is verified (no
mask), every patch is logged — far more patches per image than inpainting.

Example:
    export TMPDIR=/mnt/HDD_12TB/bam_ki/tmp
    python sr_verifier_reliability_sweep.py \
        --target_id stabilityai/stable-diffusion-x4-upscaler \
        --draft_ckpt /mnt/HDD_12TB/bam_ki/ckpt_sr/draft_sr_final.pt --use_ema_draft \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/div2k/valid \
        --out_root  /mnt/HDD_12TB/bam_ki/results/sr_reliability_div2k \
        --num_images 100 --seeds 0 1 2 3 4 5 --lr_size 128 --scale 4
"""
import argparse
import csv
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
from inference.speculative_general import fgsr_refine


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    return transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(img).unsqueeze(0)


def build_manifest(args):
    root = Path(args.data_root)
    all_imgs = []
    for e in ("jpg", "jpeg", "png", "webp"):
        all_imgs += list(root.rglob(f"*.{e}"))
    all_imgs = sorted(str(p) for p in all_imgs)
    rng = random.Random(args.seed_base)
    rng.shuffle(all_imgs)
    return [{"idx": i, "image_path": p} for i, p in enumerate(all_imgs[:args.num_images])]


def load_models(args, device):
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
        draft.load_state_dict(ck[key]); print(f"[sr-rel] loaded {key}")
    else:
        draft = DraftEpsUNet(**dk).to(device).eval()
        print("[sr-rel] no draft ckpt -> random (dummy/smoke only)")
    return target, draft, sch, DWT2D("haar").to(device)


def run_one_logged(target, draft, sch, dwt, item, seed, args, device):
    hr_size = args.lr_size * args.scale
    hr = load_image(item["image_path"], hr_size).to(device)
    lr = F.interpolate(hr, size=(args.lr_size, args.lr_size),
                       mode="bicubic", align_corners=False).clamp(-1, 1)
    cond_lr, nl = target.prepare_lr_cond(lr, noise_level=args.noise_level)
    region = torch.ones(1, 1, args.lr_size, args.lr_size, device=device)
    torch.manual_seed(seed)
    z_init = torch.randn(1, target.latent_ch, args.lr_size, args.lr_size, device=device)
    if target.available:
        cond_emb, uncond_emb, _ = target.get_text_embeddings(
            args.prompt, batch_size=1, guidance_scale=args.guidance_scale)
    else:
        cond_emb, uncond_emb = None, None
    t0 = time.perf_counter()
    with torch.no_grad():
        z_out, stats = fgsr_refine(
            target, draft, z_init, cond_lr, region, sch,
            num_inference_steps=args.num_steps, K=args.K, patch_size=args.patch,
            t_spec_start_norm=args.t_spec_start, beta=args.beta,
            tol_low=args.tol_low, tol_high=args.tol_high,
            boundary_weight=0.0, mask_interior_weight=0.0,  # SR: wavelet-only
            guidance_scale=args.guidance_scale,
            cond_emb=cond_emb, uncond_emb=uncond_emb,
            known_z=None, blend_known=False,
            blend_temperature=args.blend_temperature,
            x0_thr_strict=args.x0_thr_strict, x0_thr_loose=args.x0_thr_loose,
            x0_strict_center=args.x0_strict_center, x0_strict_width=args.x0_strict_width,
            drift_k_switch_threshold=args.drift_k_switch_threshold,
            k_switch_threshold=args.k_switch_threshold,
            dwt=dwt, collect_patch_logs=True, target_extra={"noise_level": nl})
    return stats, time.perf_counter() - t0


def main(args):
    device = torch.device(args.device)
    target, draft, sch, dwt = load_models(args, device)
    if not target.available:
        print("[sr-rel] WARNING: dummy mode, logs not meaningful")
    manifest = build_manifest(args)
    logs_dir = os.path.join(args.out_root, "patch_logs")
    os.makedirs(logs_dir, exist_ok=True)

    csv_path = os.path.join(args.out_root, "reliability_runs.csv")
    done = set()
    if args.resume and os.path.isfile(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add(r["log_name"])
    fcsv = open(csv_path, "a", newline="")
    w = csv.writer(fcsv)
    if not (args.resume and os.path.isfile(csv_path)):
        w.writerow(["log_name", "image_path", "seed", "time_sec",
                    "target_nfe", "draft_nfe", "accept_rate", "n_patches"])

    print(f"[sr-rel] {len(manifest)} images x {len(args.seeds)} seeds")
    for seed in args.seeds:
        for item in manifest:
            log_name = f"img_{item['idx']:04d}_seed_{seed}.pt"
            if log_name in done:
                continue
            real_seed = seed * 100000 + item["idx"]
            stats, t_run = run_one_logged(target, draft, sch, dwt, item, real_seed, args, device)
            pl = stats.get("patch_logs", {})
            n = int(pl["d_x0"].numel()) if "d_x0" in pl else 0
            torch.save({"patch_logs": pl,
                        "image_id": os.path.basename(item["image_path"]),
                        "seed": real_seed}, os.path.join(logs_dir, log_name))
            w.writerow([log_name, item["image_path"], real_seed, round(t_run, 4),
                        stats["target_calls"], stats["draft_calls"],
                        f"{stats['accept_rate']:.4f}", n])
            fcsv.flush()
            print(f"  {log_name}  patches={n}  acc={stats['accept_rate']:.3f}")
    fcsv.close()
    print(f"\n[sr-rel] done -> {logs_dir}")
    print(f"[sr-rel] next: python analyze_verifier_reliability.py "
          f"--logs_dir {logs_dir} --out_dir {args.out_root}/analysis")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--draft_ckpt", type=str, default="")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-x4-upscaler")
    p.add_argument("--target_dtype", type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--use_ema_draft", action="store_true")
    p.add_argument("--num_images", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--lr_size", type=int, default=128)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--noise_level", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--t_spec_start", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--tol_low", type=float, default=0.03)
    p.add_argument("--tol_high", type=float, default=0.30)
    p.add_argument("--blend_temperature", type=float, default=0.10)
    p.add_argument("--x0_thr_strict", type=float, default=0.02)
    p.add_argument("--x0_thr_loose", type=float, default=0.07)
    p.add_argument("--x0_strict_center", type=float, default=0.45)
    p.add_argument("--x0_strict_width", type=float, default=0.12)
    p.add_argument("--drift_k_switch_threshold", type=float, default=0.006)
    p.add_argument("--k_switch_threshold", type=float, default=0.60)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
