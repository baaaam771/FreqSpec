#!/usr/bin/env python
"""
analyze_perceptual.py — Inpainting-appropriate perceptual evaluation.

Motivation
----------
Inpainting is a multi-modal generation task: a masked region has many
plausible completions, not a single ground-truth answer. Reference-based
metrics (PSNR/SSIM/LPIPS against the original image, or LPIPS against
the target_s50 output) therefore mismeasure the task — they penalize
plausible-but-different completions and reward "copying the target".

This script evaluates each method on its OWN merits:
  - CLIP-IQA   : no-reference perceptual quality (is the image naturally lit,
                 sharp, well-composed?)
  - MUSIQ      : no-reference quality, deep IQA model
  - BRISQUE    : classical no-reference quality (lower = better)
  - FID        : how close is the *distribution* of method outputs to the
                 distribution of real photographs (lower = more realistic)
  - KID        : same idea, more stable for small samples (lower = better)

Reads each method's saved outputs from baseline_sweep.py — no re-inference.

Example:
    python analyze_perceptual.py \\
        --sweep_root /mnt/HDD_12TB/bam_ki/results/sweep_v2_coco_n100 \\
        --real_dir /mnt/HDD_12TB/bam_ki/datasets/coco2017/val2017 \\
        --dataset_name COCO --n_real 1000

Notes on sample size
--------------------
FID is meaningful only when both sets have enough images. n>=1000 is
standard. With n=100 method outputs vs n>=1000 real images, FID becomes
useful as a *relative* comparison between methods (the absolute number
is still biased upward, but the ranking is informative).

For n=20, FID is too noisy — only CLIP-IQA / MUSIQ / BRISQUE are
trustworthy.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ============================================================
# Helpers
# ============================================================
def list_method_dirs(sweep_root: Path):
    return sorted(d for d in sweep_root.iterdir()
                  if d.is_dir() and (d / "results.csv").is_file())


def list_method_outputs(method_dir: Path):
    """Return list of out.png paths under method_dir/img_NNN/."""
    paths = []
    for sub in sorted(method_dir.iterdir()):
        if sub.is_dir() and sub.name.startswith("img_"):
            p = sub / "out.png"
            if p.is_file():
                paths.append(p)
    return paths


def load_img_tensor(path, size=None):
    """Load image as torch tensor [1,3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ============================================================
# (A) No-reference IQA: CLIP-IQA, MUSIQ, BRISQUE
# ============================================================
def compute_iqa_scores(method_outputs, device, iqa_metrics):
    """Compute per-image IQA scores for one method. Returns {metric_name: [scores]}."""
    scores = {name: [] for name in iqa_metrics}
    for p in method_outputs:
        img = load_img_tensor(p).to(device)
        for name, fn in iqa_metrics.items():
            with torch.no_grad():
                s = fn(img).item()
            scores[name].append(s)
    return scores


def build_iqa_metrics(device):
    """Construct pyiqa metrics. Returns dict {name: callable(img) -> score}."""
    import pyiqa
    metrics = {}
    # CLIP-IQA: higher = better (perceptual quality)
    metrics["clipiqa"] = pyiqa.create_metric(
        "clipiqa", device=device, as_loss=False
    )
    # MUSIQ: higher = better, deep IQA
    metrics["musiq"] = pyiqa.create_metric(
        "musiq", device=device, as_loss=False
    )
    # BRISQUE: LOWER = better, classical no-ref
    metrics["brisque"] = pyiqa.create_metric(
        "brisque", device=device, as_loss=False
    )
    return metrics


# ============================================================
# (B) FID / KID using clean-fid
# ============================================================
def compute_fid_kid(method_dir, real_dir, n_real, device):
    """
    Compute FID + KID between method outputs and a sample of real photographs.

    Note: uses num_workers=0 to avoid Python 3.14 multiprocessing pickle errors
    (clean-fid's internal resizer is a local function and not picklable under
    forkserver, which became default in 3.14).
    """
    from cleanfid import fid as cleanfid

    # stage method outputs into a flat folder (clean-fid wants images directly)
    stage = method_dir / "_fid_stage"
    if not stage.exists():
        stage.mkdir()
        for p in list_method_outputs(method_dir):
            target = stage / f"{p.parent.name}.png"
            if not target.exists():
                target.symlink_to(p.resolve())

    score_fid = cleanfid.compute_fid(
        str(stage), str(real_dir), mode="clean", device=device,
        num_workers=0, batch_size=8,
    )
    score_kid = cleanfid.compute_kid(
        str(stage), str(real_dir), mode="clean", device=device,
        num_workers=0, batch_size=8,
    )
    return score_fid, score_kid


# ============================================================
# Main
# ============================================================
def main(args):
    sweep_root = Path(args.sweep_root)
    device = args.device

    method_dirs = list_method_dirs(sweep_root)
    print(f"[perceptual] methods: {[d.name for d in method_dirs]}")
    n_imgs = len(list_method_outputs(method_dirs[0]))
    print(f"[perceptual] n images per method: {n_imgs}")

    if args.skip_fid and args.skip_iqa:
        print("nothing to do (both --skip_fid and --skip_iqa)")
        return

    # -------- IQA (no-reference) --------
    iqa_summary = {}
    if not args.skip_iqa:
        print("\n[perceptual] building IQA models...")
        iqa = build_iqa_metrics(device)
        print(f"[perceptual] IQA metrics: {list(iqa.keys())}")
        for mdir in method_dirs:
            outs = list_method_outputs(mdir)
            print(f"[perceptual] IQA on {mdir.name} ({len(outs)} images)...")
            scores = compute_iqa_scores(outs, device, iqa)
            iqa_summary[mdir.name] = {
                k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()
            }
        # free GPU memory before FID
        del iqa
        torch.cuda.empty_cache()

    # -------- FID / KID --------
    fid_summary = {}
    if not args.skip_fid:
        assert args.real_dir, "--real_dir required for FID/KID"
        print(f"\n[perceptual] computing FID/KID against real images in "
              f"{args.real_dir}")
        for mdir in method_dirs:
            print(f"[perceptual] FID/KID on {mdir.name}...")
            f, k = compute_fid_kid(mdir, args.real_dir, args.n_real, device)
            fid_summary[mdir.name] = {"fid": float(f), "kid": float(k)}
            print(f"  -> FID={f:.3f}  KID={k:.5f}")

    # -------- print summary --------
    print(f"\n{'='*100}")
    print(f"PERCEPTUAL SUMMARY — {args.dataset_name}  "
          f"(n_imgs per method = {n_imgs})")
    print(f"{'='*100}")
    hdr = f"{'method':22}"
    if not args.skip_iqa:
        hdr += f" {'CLIP-IQA↑':>10} {'MUSIQ↑':>8} {'BRISQUE↓':>10}"
    if not args.skip_fid:
        hdr += f" {'FID↓':>8} {'KID↓':>9}"
    print(hdr)
    print("-" * len(hdr))
    # speedup-sorted (slowest first to mirror previous analyzers)
    # but here we want method order: target_s50, others by speedup desc
    method_names = sorted(set(list(iqa_summary) + list(fid_summary)))
    for m in method_names:
        line = f"{m:22}"
        if m in iqa_summary:
            s = iqa_summary[m]
            line += (f" {s['clipiqa'][0]:>10.4f}"
                     f" {s['musiq'][0]:>8.2f}"
                     f" {s['brisque'][0]:>10.3f}")
        if m in fid_summary:
            line += f" {fid_summary[m]['fid']:>8.2f} {fid_summary[m]['kid']:>9.5f}"
        print(line)

    # speed-matched pairs (re-load from previous analyzer's summary.json if exists)
    summary_json = sweep_root / "summary.json"
    if summary_json.is_file():
        with open(summary_json) as f:
            prev = json.load(f)
        print(f"\n{'='*100}")
        print("SPEED-MATCHED PERCEPTUAL COMPARISON")
        print(f"{'='*100}")
        fs_methods = [m for m in prev if m.startswith("freqspec")]
        tgt_methods = [m for m in prev if m.startswith("target")]
        for fs in fs_methods:
            sp = prev[fs]["speedup"]
            closest = min(tgt_methods,
                          key=lambda t: abs(prev[t]["speedup"] - sp))
            print(f"\n  {fs} ({sp:.2f}x) vs {closest} "
                  f"({prev[closest]['speedup']:.2f}x):")
            for m_name, prefix in [(fs, "    FreqSpec"), (closest, "    Target  ")]:
                bits = [prefix]
                if m_name in iqa_summary:
                    s = iqa_summary[m_name]
                    bits.append(f"CLIP-IQA={s['clipiqa'][0]:.4f}")
                    bits.append(f"MUSIQ={s['musiq'][0]:.2f}")
                    bits.append(f"BRISQUE={s['brisque'][0]:.2f}")
                if m_name in fid_summary:
                    bits.append(f"FID={fid_summary[m_name]['fid']:.2f}")
                    bits.append(f"KID={fid_summary[m_name]['kid']:.5f}")
                print("  ".join(bits))

    # save
    out = {"iqa": iqa_summary, "fid": fid_summary, "n_imgs": n_imgs}
    out_path = sweep_root / "perceptual_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[perceptual] saved -> {out_path}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="dataset")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--real_dir", type=str, default="",
                   help="Folder of real photographs for FID/KID. "
                        "For COCO use val2017, for Places2 use Places2 root, etc.")
    p.add_argument("--n_real", type=int, default=1000,
                   help="(Informational only — clean-fid uses all images "
                        "found under real_dir.)")
    p.add_argument("--skip_fid", action="store_true",
                   help="Skip FID/KID (useful when n_imgs is small, e.g. n=20).")
    p.add_argument("--skip_iqa", action="store_true",
                   help="Skip CLIP-IQA/MUSIQ/BRISQUE (rarely needed).")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())