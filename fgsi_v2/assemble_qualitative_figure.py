#!/usr/bin/env python
"""
assemble_qualitative_figure.py — Build a qualitative comparison grid figure
from baseline_sweep.py output.

Reads the file layout produced by baseline_sweep.py:

    {sweep_dir}/
        manifest.json
        target_s50/   results.csv  img_000/{gt,out,mask}.png  img_001/...
        target_s30/   results.csv  img_000/{gt,out,mask}.png  img_001/...
        freqspec_strict/   ...
        freqspec_default/  ...

Produces:
    {out_path}.pdf   +   {out_path}.png
    Grid: rows = samples, cols = [Masked Input, ...methods..., GT]

Usage:
    python assemble_qualitative_figure.py \\
        --sweep_dir /mnt/HDD_12TB/bam_ki/results/qualitative_coco_run \\
        --methods target_s50 target_s30 freqspec_strict freqspec_default \\
        --sample_indices 0 2 4 6 \\
        --out_path figures/qualitative_coco
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Human-readable column titles
METHOD_TITLES = {
    "target_s50":       "Target (50 steps)",
    "target_s40":       "Target (40 steps)",
    "target_s37":       "Target (37 steps)",
    "target_s30":       "Target (30 steps)",
    "target_s25":       "Target (25 steps)",
    "freqspec_strict":  "FreqSpec (strict)",
    "freqspec_mid":     "FreqSpec (mid)",
    "freqspec_default": "FreqSpec (default)",
}


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return m  # H, W in [0, 1]


def make_masked_input(gt, mask, mode="gray"):
    """Compose the input image with mask overlay for visualization."""
    if mode == "gray":
        # Replace masked region with mid-gray
        out = gt.copy().astype(np.float32)
        m3 = mask[..., None]
        out = out * (1 - m3) + 128.0 * m3
        return out.astype(np.uint8)
    elif mode == "red":
        # Red semi-transparent overlay
        out = gt.copy().astype(np.float32)
        red = np.array([255.0, 60.0, 60.0])
        alpha = 0.55
        m3 = mask[..., None]
        out = out * (1 - alpha * m3) + red[None, None, :] * (alpha * m3)
        return out.astype(np.uint8)
    elif mode == "outline":
        # White outline of mask boundary
        import scipy.ndimage as ndi
        out = gt.copy()
        boundary = ndi.binary_dilation(mask > 0.5) ^ (mask > 0.5)
        out[boundary] = [255, 255, 0]
        return out
    else:
        return gt


def find_sample_dirs(sweep_dir, method, indices):
    """Return list of (idx, sample_dir) for given method."""
    base = Path(sweep_dir) / method
    if not base.exists():
        raise FileNotFoundError(f"method dir not found: {base}")
    out = []
    for idx in indices:
        d = base / f"img_{idx:03d}"
        if not d.exists():
            raise FileNotFoundError(f"sample dir not found: {d}")
        out.append((idx, d))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True,
                   help="baseline_sweep.py --out_root path")
    p.add_argument("--methods", nargs="+", required=True,
                   help="method names to include as columns, in order")
    p.add_argument("--sample_indices", type=int, nargs="+", required=True,
                   help="manifest indices to include as rows")
    p.add_argument("--out_path", required=True,
                   help="output path WITHOUT extension; .pdf and .png written")
    p.add_argument("--mask_overlay", default="gray",
                   choices=["gray", "red", "outline"],
                   help="how to render the input column")
    p.add_argument("--include_input", action="store_true", default=True)
    p.add_argument("--include_gt", action="store_true", default=True)
    p.add_argument("--cell_size", type=float, default=2.3,
                   help="inches per cell")
    p.add_argument("--title_fontsize", type=int, default=11)
    p.add_argument("--row_label_fontsize", type=int, default=10)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show_caption_below", action="store_true",
                   help="show COCO prompt under each row")
    args = p.parse_args()

    # Load manifest for prompts / paths
    manifest_path = Path(args.sweep_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"[warn] manifest not found at {manifest_path}; prompts omitted")
        manifest = []
    else:
        manifest = json.loads(manifest_path.read_text())

    idx_to_prompt = {it["idx"]: it.get("prompt", "") for it in manifest}

    # Resolve sample dirs and collect images
    n_rows = len(args.sample_indices)
    cols = []
    if args.include_input:
        cols.append("Input")
    cols += [METHOD_TITLES.get(m, m) for m in args.methods]
    if args.include_gt:
        cols.append("Ground Truth")
    n_cols = len(cols)

    # mask/gt are shared across methods — pull from the first method's dir
    ref_method = args.methods[0]
    ref_dirs = find_sample_dirs(args.sweep_dir, ref_method, args.sample_indices)

    fig_h = args.cell_size * n_rows + (0.6 if args.show_caption_below else 0.4)
    fig_w = args.cell_size * n_cols + 0.3
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)

    for r, (idx, sample_dir) in enumerate(ref_dirs):
        # shared assets
        gt_img = load_image(sample_dir / "gt.png")
        mask = load_mask(sample_dir / "mask.png")
        # If gt and out differ in resolution from mask, resize mask
        if mask.shape != gt_img.shape[:2]:
            from PIL import Image as _PI
            mask = np.array(
                _PI.fromarray((mask * 255).astype(np.uint8))
                .resize((gt_img.shape[1], gt_img.shape[0]), _PI.NEAREST)
            ).astype(np.float32) / 255.0

        c = 0
        # Input column
        if args.include_input:
            ax = axes[r, c]
            ax.imshow(make_masked_input(gt_img, mask, mode=args.mask_overlay))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=args.title_fontsize)
            ax.set_ylabel(f"Sample {idx}",
                          fontsize=args.row_label_fontsize)
            c += 1

        # method columns
        for m in args.methods:
            out_path = Path(args.sweep_dir) / m / f"img_{idx:03d}" / "out.png"
            if not out_path.exists():
                print(f"[warn] missing: {out_path}")
                im = np.zeros_like(gt_img)
            else:
                im = load_image(out_path)
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=args.title_fontsize)
            c += 1

        # GT column
        if args.include_gt:
            ax = axes[r, c]
            ax.imshow(gt_img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=args.title_fontsize)
            c += 1

        if args.show_caption_below:
            prompt = idx_to_prompt.get(idx, "")
            if prompt:
                # Place caption under sample row (use the leftmost cell)
                axes[r, 0].set_xlabel(
                    f"\u201c{prompt}\u201d",
                    fontsize=args.row_label_fontsize - 1,
                    style="italic",
                )

    plt.subplots_adjust(wspace=0.04, hspace=0.06)

    out_path = args.out_path
    if not out_path.endswith((".pdf", ".png")):
        # write both extensions
        plt.savefig(out_path + ".pdf", bbox_inches="tight", dpi=args.dpi)
        plt.savefig(out_path + ".png", bbox_inches="tight", dpi=args.dpi)
        print(f"[done] saved {out_path}.pdf and {out_path}.png")
    else:
        plt.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
        # also write companion
        other = out_path.rsplit(".", 1)[0] + (".png" if out_path.endswith(".pdf") else ".pdf")
        plt.savefig(other, bbox_inches="tight", dpi=args.dpi)
        print(f"[done] saved {out_path} and {other}")


if __name__ == "__main__":
    main()
