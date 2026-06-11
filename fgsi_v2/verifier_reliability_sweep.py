#!/usr/bin/env python
"""
verifier_reliability_sweep.py  —  collect per-patch verifier logs (Table A).

Runs FreqSpec (Combo 2) on a fixed, seed-deterministic manifest with
`collect_patch_logs=True`, and dumps one .pt log per image-seed containing the
per-mask-interior-patch quantities needed for risk-coverage / AURC / FAR /
accepted-rejected analysis:
    d_x0, s_eps, w, saliency, t_norm   (+ image_id, seed)

This is a thin driver: it reuses the existing model loading, preprocessing,
manifest building and latent preparation from baseline_sweep.py, so the runs
are paired and fair with the rest of the paper's evaluation. It does NOT add
any new inference logic — only the logging flag already patched into
fgsr_inpaint.

After running, analyze with:
    python analyze_verifier_reliability.py --logs_dir <out>/patch_logs --out_dir <out>/analysis

Example (COCO, Combo 2, 200 images):
    python verifier_reliability_sweep.py \\
        --target_id /mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \\
        --draft_ckpt /mnt/HDD_12TB/bam_ki/runs/sdxl_coco/draft_final.pt --use_ema_draft \\
        --data_root /mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017 \\
        --caption_json /mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json \\
        --out_root /mnt/HDD_12TB/bam_ki/results/verifier_coco \\
        --num_images 200 --image_size 1024 \\
        --x0_thr_strict 0.02 --x0_thr_loose 0.07 \\
        --x0_strict_center 0.45 --x0_strict_width 0.12 \\
        --blend_temperature 0.10 --mask_interior_weight 0.5 \\
        --drift_k_switch_threshold 0.006

For paired multi-seed evaluation (200 images x 3 seeds), pass --seeds 42 43 44.
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.target_wrapper import TargetWrapper
from models.draft import DraftEpsUNet
from models.wavelet import DWT2D
from training.scheduler import DDPMSchedule
from inference.speculative import fgsr_inpaint

# Reuse the exact preprocessing / manifest / timing helpers from the sweep.
from baseline_sweep import (
    load_image, make_mask, build_manifest, timed_run,
    _prepare_latents, _get_emb, print_combo2_status,
)


def load_models(args, device):
    target_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                    "fp32": torch.float32}[args.target_dtype]
    target = TargetWrapper(model_id=args.target_id, device=device,
                           dtype=target_dtype)
    assert target.available, "Target model must be available for real eval."
    sch = DDPMSchedule(
        num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
        beta_start=target.scheduler_ref.config.beta_start,
        beta_end=target.scheduler_ref.config.beta_end,
        beta_schedule=target.scheduler_ref.config.beta_schedule,
        device=device,
    )
    draft_kwargs = {"latent_ch": target.latent_ch,
                    "num_train_timesteps": sch.num_train_timesteps}
    ck = torch.load(args.draft_ckpt, map_location=device)
    saved_args = ck.get("args", {})
    if "draft_base_ch" in saved_args:
        draft_kwargs["base_ch"] = saved_args["draft_base_ch"]
        draft_kwargs["ch_mult"] = tuple(saved_args["draft_ch_mult"])
        draft_kwargs["t_dim"] = saved_args["draft_t_dim"]
    draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
    if args.use_ema_draft and ck.get("ema_draft") is not None:
        draft.load_state_dict(ck["ema_draft"]); print("[verif] loaded EMA draft")
    else:
        draft.load_state_dict(ck["draft"]); print("[verif] loaded draft")
    print(f"[verif] draft params: "
          f"{sum(p.numel() for p in draft.parameters())/1e6:.2f}M")
    dwt = DWT2D("haar").to(device)
    return target, draft, sch, dwt


def run_one_logged(target, draft, sch, dwt, item, args, device):
    img, mask_pix, z0, mask_z, cond_z, z_init = _prepare_latents(
        target, sch, item, args, device)
    cond_emb, uncond_emb = _get_emb(target, item, args, z0)
    with torch.no_grad():
        (z_out, stats), t_run = timed_run(
            lambda: fgsr_inpaint(
                target, draft, z_init.clone(), cond_z, mask_z, sch,
                num_inference_steps=args.num_steps,
                K=args.K, patch_size=args.patch,
                t_spec_start_norm=args.t_spec_start, beta=args.beta,
                tol_low=args.tol_low, tol_high=args.tol_high,
                boundary_weight=args.boundary_weight,
                mask_interior_weight=args.mask_interior_weight,
                uniform_saliency=False,
                dwt=dwt, verbose=False,
                guidance_scale=args.guidance_scale,
                cond_emb=cond_emb, uncond_emb=uncond_emb,
                known_z=z0, blend_known=True,
                k_switch_threshold=args.k_switch_threshold,
                blend_temperature=args.blend_temperature,
                x0_thr_strict=args.x0_thr_strict,
                x0_thr_loose=args.x0_thr_loose,
                x0_strict_center=args.x0_strict_center,
                x0_strict_width=args.x0_strict_width,
                drift_k_switch_threshold=args.drift_k_switch_threshold,
                collect_patch_logs=True,
            ),
            device,
        )
    return stats, t_run


def main(args):
    device = torch.device(args.device)
    target, draft, sch, dwt = load_models(args, device)

    base_manifest = build_manifest(args)
    os.makedirs(args.out_root, exist_ok=True)
    logs_dir = os.path.join(args.out_root, "patch_logs")
    os.makedirs(logs_dir, exist_ok=True)

    print(f"[verif] {len(base_manifest)} images x {len(args.seeds)} seeds")
    print_combo2_status(_as_sweep_args(args))

    csv_path = os.path.join(args.out_root, "verifier_runs.csv")
    done = set()
    if args.resume and os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(row["log_name"])
    write_header = not (args.resume and os.path.isfile(csv_path))
    f_csv = open(csv_path, "a", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["log_name", "image_path", "seed", "time_sec",
                         "target_nfe", "draft_nfe", "accept_rate",
                         "n_logged_patches"])

    # warmup (excluded)
    wi = dict(base_manifest[0]); wi["seed"] = args.seeds[0] * 100000
    _prepare_latents(target, sch, wi, args, device)

    for seed in args.seeds:
        for item in base_manifest:
            it = dict(item)
            it["seed"] = seed * 100000 + it["idx"]
            log_name = f"img_{it['idx']:04d}_seed_{seed}.pt"
            if log_name in done:
                continue
            stats, t_run = run_one_logged(target, draft, sch, dwt, it, args, device)
            pl = stats.get("patch_logs", {})
            n_logged = int(pl["d_x0"].numel()) if "d_x0" in pl else 0
            torch.save({"patch_logs": pl,
                        "image_id": os.path.basename(it["image_path"]),
                        "seed": it["seed"]},
                       os.path.join(logs_dir, log_name))
            writer.writerow([log_name, it["image_path"], it["seed"],
                             round(t_run, 4), stats["target_calls"],
                             stats["draft_calls"],
                             f"{stats['accept_rate']:.4f}", n_logged])
            f_csv.flush()
            print(f"  [{log_name}] tgt_nfe={stats['target_calls']} "
                  f"patches={n_logged} t={t_run:.2f}s")
    f_csv.close()
    print(f"\n[verif] done -> {logs_dir}")
    print(f"[verif] next: python analyze_verifier_reliability.py "
          f"--logs_dir {logs_dir} --out_dir {args.out_root}/analysis")


class _as_sweep_args:
    """Adapter so print_combo2_status (from baseline_sweep) can read our args."""
    def __init__(self, a):
        self.x0_threshold = None
        self.blend_temperature = a.blend_temperature
        self.x0_strict_center = a.x0_strict_center
        self.x0_thr_strict = a.x0_thr_strict
        self.x0_thr_loose = a.x0_thr_loose
        self.x0_strict_width = a.x0_strict_width
        self.mask_interior_weight = a.mask_interior_weight
        self.drift_k_switch_threshold = a.drift_k_switch_threshold
        self.k_switch_threshold = a.k_switch_threshold
        self.save_usage_maps = False


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target_id", required=True)
    p.add_argument("--draft_ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--caption_json", default="")
    p.add_argument("--num_images", type=int, default=200)
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for the image manifest (shared with sweep).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Diffusion/mask seeds. Use e.g. 42 43 44 for paired "
                        "multi-seed evaluation (200 images x 3 seeds).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--target_dtype", default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--auto_prompt", action="store_true")
    p.add_argument("--use_ema_draft", action="store_true")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--t_spec_start", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--boundary_weight", type=float, default=1.0)
    # Combo 2 defaults wired to the canonical full-system setting
    p.add_argument("--tol_low", type=float, default=0.03)
    p.add_argument("--tol_high", type=float, default=0.30)
    p.add_argument("--mask_interior_weight", type=float, default=0.5)
    p.add_argument("--k_switch_threshold", type=float, default=0.60)
    p.add_argument("--blend_temperature", type=float, default=0.10)
    p.add_argument("--x0_thr_strict", type=float, default=0.02)
    p.add_argument("--x0_thr_loose", type=float, default=0.07)
    p.add_argument("--x0_strict_center", type=float, default=0.45)
    p.add_argument("--x0_strict_width", type=float, default=0.12)
    p.add_argument("--drift_k_switch_threshold", type=float, default=0.006)
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
