#!/usr/bin/env python
"""
run_target_reference.py  —  50-step target reference for LPIPS_t.

Produces the full-quality target_s50 output for every image in the SHARED
manifest (same build_manifest / _prepare_latents as the sweeps, so masks, seeds
and prompts are paired). Saves out.png / gt.png / mask.png per image under
<out_root>/target_s50/img_XXX/, which is exactly the reference layout that
assemble_table_b.py and analyze_speed_matched.py expect.

This is intentionally minimal (only the 50-step target, no FreqSpec passes) so
it costs one target run per image and nothing else.

Example:
    python run_target_reference.py \\
        --target_id /mnt/HDD_12TB/bam_ki/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1 \\
        --draft_ckpt /mnt/HDD_12TB/bam_ki/runs/sdxl_coco_v2/draft_final.pt --use_ema_draft \\
        --data_root /mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017 \\
        --caption_json /mnt/HDD_12TB/bam_ki/datasets/coco2017/annotations/captions_val2017.json \\
        --out_root /mnt/HDD_12TB/bam_ki/results/saliency_ablation_coco \\
        --num_images 300 --image_size 1024
"""
import argparse
import csv
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.speculative import baseline_inpaint
from baseline_sweep import (
    build_manifest, timed_run, _prepare_latents, _get_emb, save_rgb, save_gray,
)
from verifier_reliability_sweep import load_models


def main(args):
    device = torch.device(args.device)
    target, draft, sch, dwt = load_models(args, device)  # draft unused but keeps loader shared
    manifest = build_manifest(args)
    ref_dir = os.path.join(args.out_root, "target_s50")
    os.makedirs(ref_dir, exist_ok=True)
    csv_path = os.path.join(ref_dir, "results.csv")

    done = set()
    if args.resume and os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(int(row["idx"]))
    write_header = not (args.resume and os.path.isfile(csv_path))
    f_csv = open(csv_path, "a", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["idx", "image_path", "method", "num_steps",
                         "time_sec", "target_nfe", "draft_nfe", "accept_rate"])

    # warmup
    _prepare_latents(target, sch, manifest[0], args, device)

    print(f"[ref] target_s50 over {len(manifest)} images -> {ref_dir}")
    for item in manifest:
        if item["idx"] in done:
            continue
        img, mask_pix, z0, mask_z, cond_z, z_init = _prepare_latents(
            target, sch, item, args, device)
        cond_emb, uncond_emb = _get_emb(target, item, args, z0)
        with torch.no_grad():
            (z_out, stats), t_run = timed_run(
                lambda: baseline_inpaint(
                    target, z_init.clone(), cond_z, mask_z, sch,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    cond_emb=cond_emb, uncond_emb=uncond_emb,
                    known_z=z0, blend_known=True),
                device)
        out = target.decode_latent(z_out)
        out = img * (1 - mask_pix) + out * mask_pix
        d = os.path.join(ref_dir, f"img_{item['idx']:03d}")
        os.makedirs(d, exist_ok=True)
        save_rgb(img, os.path.join(d, "gt.png"))
        save_rgb(out, os.path.join(d, "out.png"))
        save_gray(mask_pix, os.path.join(d, "mask.png"))
        writer.writerow([item["idx"], item["image_path"], "target_s50", 50,
                         round(t_run, 4), stats["target_calls"], 0, ""])
        f_csv.flush()
        print(f"  [target_s50] img {item['idx']:03d} t={t_run:.2f}s")
    f_csv.close()
    print(f"[ref] done -> {ref_dir}")


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
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
