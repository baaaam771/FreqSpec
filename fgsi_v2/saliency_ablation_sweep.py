#!/usr/bin/env python
"""
saliency_ablation_sweep.py  —  Table B (ablation of saliency signals).

Runs FreqSpec on a single fixed manifest under several saliency configurations,
holding EVERYTHING else fixed (same draft, images, masks, prompts, seeds, K,
tolerances, soft blending, x0 gate). For each configuration it:
  * saves out.png / gt.png / mask.png  -> feed to the existing compute_metrics.py
    / analyze_speed_matched.py for (mask) LPIPS, boundary LPIPS, speedup;
  * saves per-image patch logs          -> feed to analyze_verifier_reliability.py
    for coverage-matched False-Accept and AURC of that saliency signal.

Saliency configurations (Table B rows):
    none_uniform       uniform tolerance (no saliency)
    random             random saliency map (control)
    sobel              Sobel gradient magnitude
    variance           local latent variance
    laplacian          Laplacian energy
    wavelet_only       Haar high-frequency energy only
    boundary_only      mask-boundary term only
    wavelet_boundary   wavelet + boundary
    full               wavelet + boundary + mask-interior (paper default)

The decisive comparisons the reviewers asked for:
    wavelet_only vs sobel/variance/laplacian  -> is "frequency-guided" justified?
    wavelet_only vs wavelet_boundary          -> does the boundary term help?
    wavelet_boundary vs full                   -> does mask-interior help?

Two operating modes are recommended (run the script twice):
  (a) fixed hyperparameters  -> realistic operating point (default below);
  (b) coverage-matched       -> pass --tol_low/--tol_high so each config lands
      near 50% coverage, isolating selection quality from selection quantity.

Example:
    python saliency_ablation_sweep.py \\
        --target_id .../stable-diffusion-xl-1.0-inpainting-0.1 \\
        --draft_ckpt .../draft_final.pt --use_ema_draft \\
        --data_root .../coco2017/val2017 \\
        --caption_json .../annotations/captions_val2017.json \\
        --out_root .../saliency_ablation_coco \\
        --num_images 300 --image_size 1024
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.speculative import fgsr_inpaint
from baseline_sweep import (
    build_manifest, timed_run, _prepare_latents, _get_emb, save_rgb, save_gray,
)
from verifier_reliability_sweep import load_models


# Each config sets: uniform_saliency, saliency_signal, saliency_use_base,
# boundary_weight, mask_interior_weight.
def build_saliency_configs(args):
    bw = args.boundary_weight
    iw = args.mask_interior_weight
    return [
        # name              uniform signal       use_base  bound  interior
        ("none_uniform",    True,  "uniform",    True,     0.0,   0.0),
        ("random",          False, "random",     True,     0.0,   0.0),
        ("sobel",           False, "sobel",      True,     0.0,   0.0),
        ("variance",        False, "variance",   True,     0.0,   0.0),
        ("laplacian",       False, "laplacian",  True,     0.0,   0.0),
        ("wavelet_only",    False, "wavelet",    True,     0.0,   0.0),
        ("boundary_only",   False, "wavelet",    False,    bw,    0.0),
        ("wavelet_boundary",False, "wavelet",    True,     bw,    0.0),
        ("full",            False, "wavelet",    True,     bw,    iw),
    ]


def run_one(target, draft, sch, dwt, item, cfg, args, device):
    name, uniform, signal, use_base, bound_w, inter_w = cfg
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
                boundary_weight=bound_w,
                mask_interior_weight=inter_w,
                uniform_saliency=uniform,
                saliency_signal=signal,
                saliency_use_base=use_base,
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
    # outputs for the existing LPIPS / boundary-LPIPS tooling
    m_out = os.path.join(args.out_root, name, f"img_{item['idx']:03d}")
    os.makedirs(m_out, exist_ok=True)
    out = target.decode_latent(z_out)
    out = img * (1 - mask_pix) + out * mask_pix
    save_rgb(img, os.path.join(m_out, "gt.png"))
    save_rgb(out, os.path.join(m_out, "out.png"))
    save_gray(mask_pix, os.path.join(m_out, "mask.png"))
    # patch logs for FAR / AURC analysis
    logs_dir = os.path.join(args.out_root, name, "patch_logs")
    os.makedirs(logs_dir, exist_ok=True)
    pl = stats.get("patch_logs", {})
    torch.save({"patch_logs": pl,
                "image_id": os.path.basename(item["image_path"]),
                "seed": item["seed"]},
               os.path.join(logs_dir, f"img_{item['idx']:03d}.pt"))
    n_logged = int(pl["d_x0"].numel()) if "d_x0" in pl else 0
    return [item["idx"], name, round(t_run, 4), stats["target_calls"],
            stats["draft_calls"], f"{stats['accept_rate']:.4f}", n_logged]


def main(args):
    device = torch.device(args.device)
    target, draft, sch, dwt = load_models(args, device)
    manifest = build_manifest(args)
    configs = build_saliency_configs(args)
    os.makedirs(args.out_root, exist_ok=True)

    print(f"[saliency] {len(manifest)} images x {len(configs)} configs")
    print(f"[saliency] configs: {[c[0] for c in configs]}")

    # warmup
    _prepare_latents(target, sch, manifest[0], args, device)

    for cfg in configs:
        name = cfg[0]
        csv_path = os.path.join(args.out_root, f"{name}_runs.csv")
        done = set()
        if args.resume and os.path.isfile(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    done.add(int(row["idx"]))
        write_header = not (args.resume and os.path.isfile(csv_path))
        f_csv = open(csv_path, "a", newline="")
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["idx", "config", "time_sec", "target_nfe",
                             "draft_nfe", "accept_rate", "n_logged_patches"])
        print(f"\n[saliency] === {name} ===")
        for item in manifest:
            if item["idx"] in done:
                continue
            row = run_one(target, draft, sch, dwt, item, cfg, args, device)
            writer.writerow(row); f_csv.flush()
            print(f"  [{name}] img {item['idx']:04d} "
                  f"tgt_nfe={row[3]} acc={row[5]} patches={row[6]}")
        f_csv.close()

    print(f"\n[saliency] done -> {args.out_root}")
    print("[saliency] per-config analysis:")
    print("  LPIPS / bLPIPS / speedup : compute_metrics.py + analyze_speed_matched.py")
    print("  FAR / AURC               : analyze_verifier_reliability.py "
          "--logs_dir <out>/<config>/patch_logs")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target_id", required=True)
    p.add_argument("--draft_ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--caption_json", default="")
    p.add_argument("--num_images", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--mask_interior_weight", type=float, default=0.5)
    # held-fixed Combo 2 calibration (same for every saliency config)
    p.add_argument("--tol_low", type=float, default=0.03)
    p.add_argument("--tol_high", type=float, default=0.30)
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