#!/usr/bin/env python
"""
assemble_qualitative_3datasets.py — Build the main qualitative figure
(3 datasets × 8 columns) from three baseline_sweep.py runs.

Layout (per row = one dataset):
  Col 1: Masked input
  Col 2: Mask
  Col 3: Target (50 steps)
  Col 4: Target (30 steps)
  Col 5: FreqSpec strict
  Col 6: FreqSpec default
  Col 7: Draft usage map (or fallback: gray placeholder + "n/a")
  Col 8: Zoom crop (auto from mask center, taken from FreqSpec default)

Required directory layout for each --{ffhq,places2,coco}_dir:
    sweep_dir/
        manifest.json
        target_s50/      img_NNN/{gt,out,mask}.png
        target_s30/      img_NNN/{gt,out,mask}.png
        freqspec_strict/ img_NNN/{gt,out,mask}.png
        freqspec_default/img_NNN/{gt,out,mask}.png

Optional draft-usage maps:
    --usage_map_dir <dir>/
        {ffhq,places2,coco}_img_NNN_usage.png  (grayscale, larger = more draft)

Usage:
    python assemble_qualitative_3datasets.py \\
        --ffhq_dir    /path/to/qualitative_ffhq_run \\
        --places2_dir /path/to/qualitative_places2_run \\
        --coco_dir    /path/to/qualitative_coco_run \\
        --ffhq_idx 3 --places2_idx 7 --coco_idx 5 \\
        --out_path figures/fig5_qualitative_3datasets \\
        --usage_map_dir figures/usage_maps
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------
# Image utilities
# --------------------------------------------------------------------
def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return m


def make_masked_input(gt, mask, mode="gray"):
    out = gt.copy().astype(np.float32)
    m3 = mask[..., None]
    if mode == "gray":
        out = out * (1 - m3) + 128.0 * m3
    elif mode == "white":
        out = out * (1 - m3) + 255.0 * m3
    elif mode == "checker":
        H, W = gt.shape[:2]
        check = np.indices((H, W)).sum(0) // 32
        check_img = ((check % 2) * 100 + 110)[..., None]
        out = out * (1 - m3) + check_img * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def find_mask_bbox(mask, min_size=128):
    """Return (y0, y1, x0, x1) bounding box of the mask, padded to min_size."""
    H, W = mask.shape
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        # no mask; center crop
        y0, y1 = H // 2 - min_size // 2, H // 2 + min_size // 2
        x0, x1 = W // 2 - min_size // 2, W // 2 + min_size // 2
    else:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        # pad to make at least min_size square
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        half = max((y1 - y0) // 2, (x1 - x0) // 2, min_size // 2) + 12
        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)
    return int(y0), int(y1), int(x0), int(x1)


def make_zoom(img, mask, size=256):
    """Crop a zoom from the mask center, resized to (size, size)."""
    y0, y1, x0, x1 = find_mask_bbox(mask, min_size=size)
    crop = img[y0:y1, x0:x1]
    crop_pil = Image.fromarray(crop)
    crop_pil = crop_pil.resize((size, size), Image.LANCZOS)
    return np.array(crop_pil), (y0, y1, x0, x1)


def draw_zoom_rect(ax, bbox, img_shape, color="yellow", lw=2.0):
    """Overlay a rectangle on the source image showing where the zoom came from."""
    y0, y1, x0, x1 = bbox
    H, W = img_shape[:2]
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)


def load_usage_map(usage_map_dir, dataset, idx, sweep_dir=None):
    """Try to load a draft usage map for (dataset, idx).

    Search order:
        1. usage_map_dir/{dataset}_img_NNN_usage.png  (canonical)
        2. usage_map_dir/{dataset}_NNN_usage.png      (short form)
        3. usage_map_dir/{dataset}/img_NNN_usage.png  (subdir form)
        4. sweep_dir/freqspec_default/img_NNN/usage_map.png
           (auto-discovered if baseline_sweep was run with --save_usage_maps)
    """
    candidates = []
    if usage_map_dir:
        candidates += [
            Path(usage_map_dir) / f"{dataset}_img_{idx:03d}_usage.png",
            Path(usage_map_dir) / f"{dataset}_{idx:03d}_usage.png",
            Path(usage_map_dir) / dataset / f"img_{idx:03d}_usage.png",
        ]
    if sweep_dir is not None:
        candidates.append(
            Path(sweep_dir) / "freqspec_default" / f"img_{idx:03d}" / "usage_map.png"
        )
    for c in candidates:
        if c.exists():
            arr = np.array(Image.open(c).convert("L")).astype(np.float32) / 255.0
            return arr
    return None


def render_usage_panel(ax, usage_map, mask):
    """Render the usage map with a colormap; mask region only."""
    if usage_map is None:
        # placeholder: hatched gray panel
        H, W = mask.shape
        ax.imshow(np.full((H, W, 3), 200, dtype=np.uint8))
        ax.text(0.5, 0.5, "draft usage\nmap n/a",
                ha="center", va="center", fontsize=10,
                transform=ax.transAxes, color="#555",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="#777"))
    else:
        # Apply colormap (viridis); show only inside the mask
        cm = plt.cm.viridis
        rgb = cm(usage_map)[..., :3]  # H,W,3 in [0,1]
        m3 = mask[..., None]
        # outside-mask = grayscale of mask outline = light gray
        bg = np.full(rgb.shape, 0.85, dtype=np.float32)
        composed = bg * (1 - m3) + rgb * m3
        ax.imshow(composed)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
DATASETS = ["ffhq", "places2", "coco"]
DATASET_TITLE = {"ffhq": "FFHQ", "places2": "Places2", "coco": "COCO"}

COL_TITLES = [
    "Masked input",
    "Mask",
    "Target (50)",
    "Target (30)",
    "FreqSpec strict",
    "FreqSpec default",
    "Draft usage map",
    "Zoom (mask region)",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ffhq_dir", required=True)
    p.add_argument("--places2_dir", required=True)
    p.add_argument("--coco_dir", required=True)
    p.add_argument("--ffhq_idx", type=int, required=True)
    p.add_argument("--places2_idx", type=int, required=True)
    p.add_argument("--coco_idx", type=int, required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--usage_map_dir", default="",
                   help="Optional dir containing precomputed draft usage "
                        "maps (PNG). See header for naming conventions.")
    p.add_argument("--mask_overlay", default="gray",
                   choices=["gray", "white", "checker"])
    p.add_argument("--zoom_size", type=int, default=320)
    p.add_argument("--zoom_from", default="freqspec_default",
                   choices=["target_s50", "target_s30",
                            "freqspec_strict", "freqspec_default", "gt"])
    p.add_argument("--cell_size", type=float, default=2.1)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--draw_zoom_rect", action="store_true", default=True,
                   help="Draw a yellow rectangle on the Zoom-source column.")
    args = p.parse_args()

    sweep_dirs = {
        "ffhq": args.ffhq_dir,
        "places2": args.places2_dir,
        "coco": args.coco_dir,
    }
    sweep_indices = {
        "ffhq": args.ffhq_idx,
        "places2": args.places2_idx,
        "coco": args.coco_idx,
    }

    # Load all images per dataset
    rows = []
    for ds in DATASETS:
        sweep = Path(sweep_dirs[ds])
        idx = sweep_indices[ds]
        # required method dirs
        sample = {
            "dataset": ds,
            "idx": idx,
        }
        ref_dir = sweep / "freqspec_default" / f"img_{idx:03d}"
        if not ref_dir.exists():
            raise FileNotFoundError(f"{ref_dir} does not exist")
        sample["gt"] = load_image(ref_dir / "gt.png")
        sample["mask"] = load_mask(ref_dir / "mask.png")
        # Methods
        for meth in ["target_s50", "target_s30",
                     "freqspec_strict", "freqspec_default"]:
            mdir = sweep / meth / f"img_{idx:03d}"
            if not mdir.exists():
                raise FileNotFoundError(f"{mdir} does not exist")
            sample[meth] = load_image(mdir / "out.png")
        # Resize mask if shape mismatch
        if sample["mask"].shape != sample["gt"].shape[:2]:
            sample["mask"] = np.array(
                Image.fromarray((sample["mask"] * 255).astype(np.uint8))
                .resize((sample["gt"].shape[1], sample["gt"].shape[0]),
                        Image.NEAREST)
            ).astype(np.float32) / 255.0
        # Usage map (try shared dir first, then auto-discover in sweep dir)
        sample["usage"] = load_usage_map(args.usage_map_dir, ds, idx,
                                         sweep_dir=sweep)
        if sample["usage"] is not None and \
           sample["usage"].shape != sample["gt"].shape[:2]:
            sample["usage"] = np.array(
                Image.fromarray((sample["usage"] * 255).astype(np.uint8))
                .resize((sample["gt"].shape[1], sample["gt"].shape[0]),
                        Image.BILINEAR)
            ).astype(np.float32) / 255.0
        rows.append(sample)

    n_rows = len(rows)
    n_cols = len(COL_TITLES)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(args.cell_size * n_cols + 0.5,
                                       args.cell_size * n_rows + 0.5),
                              squeeze=False)

    for r, sample in enumerate(rows):
        gt = sample["gt"]; mask = sample["mask"]
        masked_input = make_masked_input(gt, mask, mode=args.mask_overlay)
        mask_rgb = (np.stack([mask] * 3, axis=-1) * 255).astype(np.uint8)

        # Choose zoom source
        zoom_src_img = sample.get(args.zoom_from, gt)
        zoom_img, zoom_bbox = make_zoom(zoom_src_img, mask, size=args.zoom_size)

        panels = [
            ("masked",       masked_input),
            ("mask",         mask_rgb),
            ("target_s50",   sample["target_s50"]),
            ("target_s30",   sample["target_s30"]),
            ("strict",       sample["freqspec_strict"]),
            ("default",      sample["freqspec_default"]),
            ("usage",        None),  # special render
            ("zoom",         zoom_img),
        ]

        zoom_src_col = {
            "target_s50": 2, "target_s30": 3,
            "freqspec_strict": 4, "freqspec_default": 5,
        }.get(args.zoom_from, 5)

        for c, (kind, im) in enumerate(panels):
            ax = axes[r, c]
            if kind == "usage":
                render_usage_panel(ax, sample["usage"], sample["mask"])
            else:
                ax.imshow(im)
                if (kind not in {"zoom"}) and args.draw_zoom_rect and c == zoom_src_col:
                    draw_zoom_rect(ax, zoom_bbox, gt.shape, color="yellow", lw=2.0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(DATASET_TITLE[sample["dataset"]],
                              fontsize=11, fontweight="bold")

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    # Add usage colorbar at bottom-right if any sample has a usage map
    has_usage = any(s.get("usage") is not None for s in rows)
    if has_usage:
        # legend strip placed outside the grid via figure-level text
        cax = fig.add_axes([0.91, 0.13, 0.012, 0.20])
        cb_data = np.linspace(0, 1, 256).reshape(-1, 1)
        cax.imshow(cb_data, aspect="auto", cmap="viridis",
                   extent=[0, 1, 0, 1], origin="lower")
        cax.set_xticks([])
        cax.set_yticks([0, 1])
        cax.set_yticklabels(["target", "draft"], fontsize=8)
        cax.set_title("$w(p)$", fontsize=9)

    out = args.out_path
    if not out.endswith((".pdf", ".png")):
        plt.savefig(out + ".pdf", bbox_inches="tight", dpi=args.dpi)
        plt.savefig(out + ".png", bbox_inches="tight", dpi=args.dpi)
        print(f"[done] saved {out}.pdf and {out}.png")
    else:
        plt.savefig(out, bbox_inches="tight", dpi=args.dpi)
        other = out.rsplit(".", 1)[0] + (".png" if out.endswith(".pdf") else ".pdf")
        plt.savefig(other, bbox_inches="tight", dpi=args.dpi)
        print(f"[done] saved {out} and {other}")


if __name__ == "__main__":
    main()
