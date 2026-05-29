#!/usr/bin/env python
"""
make_comparison_grid.py — Build side-by-side visual comparison grids.

For each chosen image, produces one grid PNG that shows all methods in a row
so you can visually judge how plausible each completion looks (instead of
relying only on numbers).

Layout per image (example):
    [GT]  [Mask] | [target_s50] [target_s40] [target_s37] [target_s30] [target_s25] | [fs_strict] [fs_mid] [fs_default]

Each method cell shows the inpainting output + a small label with speedup.

Usage:
    # default: 10 images, sweep_v2_coco_n100
    python make_comparison_grid.py \\
        --sweep_root /mnt/HDD_12TB/bam_ki/results/sweep_v2_coco_n100 \\
        --out_dir /mnt/HDD_12TB/bam_ki/results/sweep_v2_coco_n100/comparison_grids \\
        --n_grids 10

    # specify which images by index
    python make_comparison_grid.py \\
        --sweep_root .../sweep_v2_coco_n100 \\
        --out_dir .../grids \\
        --indices 0 5 10 27 42

    # diverse selection (max-divergence images, where methods differ most)
    python make_comparison_grid.py \\
        --sweep_root .../sweep_v2_coco_n100 \\
        --out_dir .../grids \\
        --pick diverse --n_grids 10
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Helpers
# ============================================================
METHOD_ORDER_PREFERENCE = [
    "target_s50", "target_s41", "target_s40", "target_s37",
    "target_s31", "target_s30", "target_s25",
    "freqspec_strict", "freqspec_mid", "freqspec_default",
]


def order_methods(method_dirs):
    """Sort methods by preferred order (targets first, freqspec last)."""
    name_to_dir = {d.name: d for d in method_dirs}
    ordered = []
    for name in METHOD_ORDER_PREFERENCE:
        if name in name_to_dir:
            ordered.append(name_to_dir[name])
    # any leftovers appended
    for d in method_dirs:
        if d not in ordered:
            ordered.append(d)
    return ordered


def load_png(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return img


def try_font(size=18):
    """Try to load a readable font; fall back to default if not found."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def label_image(img, label, font, pad=4):
    """Return image with a label bar above it."""
    w, h = img.size
    bar_h = font.size + 2 * pad
    bg = Image.new("RGB", (w, h + bar_h), color=(245, 245, 245))
    bg.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(bg)
    draw.text((pad, pad), label, fill=(20, 20, 20), font=font)
    return bg


def best_label(method_name, summary):
    """Compose '<method>\\n<speedup>x  <quick metric>' label."""
    s = summary.get(method_name, {})
    sp = s.get("speedup")
    sp_str = f"{sp:.2f}x" if sp is not None else "?"
    return f"{method_name}\n{sp_str}"


# ============================================================
# Selecting which images to grid
# ============================================================
def pick_first(n, max_idx):
    return list(range(min(n, max_idx)))


def pick_diverse(sweep_root, n, fs_method="freqspec_default",
                 ref_method="target_s40"):
    """
    Pick images where FreqSpec and reduced-step target produce most different
    outputs (highest target-deviation). Useful to see the cases where the
    methods actually disagree.
    Falls back to first-N if summary.json not present.
    """
    sj = sweep_root / "summary.json"
    if not sj.is_file():
        print("[grid] summary.json not found — falling back to first-N pick")
        return None
    # We need PER-IMAGE divergence, but summary.json only has means.
    # So compute per-image LPIPS between two methods directly.
    fs_dir = sweep_root / fs_method
    ref_dir = sweep_root / ref_method
    if not (fs_dir.is_dir() and ref_dir.is_dir()):
        return None
    import lpips, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="alex").to(device).eval()

    def img_to_t(p):
        a = np.array(Image.open(p).convert("RGB").resize((512, 512),
                                                          Image.BILINEAR))
        t = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    scores = []
    for img_subdir in sorted(fs_dir.iterdir()):
        if not (img_subdir.is_dir() and img_subdir.name.startswith("img_")):
            continue
        idx = int(img_subdir.name.split("_")[1])
        fs_p = img_subdir / "out.png"
        ref_p = ref_dir / img_subdir.name / "out.png"
        if not (fs_p.exists() and ref_p.exists()):
            continue
        with torch.no_grad():
            s = float(lp(img_to_t(fs_p), img_to_t(ref_p)).item())
        scores.append((s, idx))
    scores.sort(reverse=True)  # most divergent first
    return [idx for _, idx in scores[:n]]


# ============================================================
# Building one grid
# ============================================================
def build_grid(sweep_root, method_dirs, idx, summary, cell_size, font,
               include_gt=True, include_mask=True):
    """Produce one PIL image: GT | Mask | method_1 | method_2 | ..."""
    cells = []

    # GT and mask (any method dir has them)
    sample_dir = method_dirs[0] / f"img_{idx:03d}"
    if include_gt and (sample_dir / "gt.png").is_file():
        gt = load_png(sample_dir / "gt.png", size=cell_size)
        cells.append(label_image(gt, "Ground Truth", font))
    if include_mask and (sample_dir / "mask.png").is_file():
        m = Image.open(sample_dir / "mask.png").convert("L").resize(
            (cell_size, cell_size), Image.NEAREST
        ).convert("RGB")
        cells.append(label_image(m, "Mask", font))

    # method outputs
    for mdir in method_dirs:
        out_p = mdir / f"img_{idx:03d}" / "out.png"
        if not out_p.is_file():
            continue
        img = load_png(out_p, size=cell_size)
        label = best_label(mdir.name, summary)
        cells.append(label_image(img, label, font))

    # tile horizontally
    if not cells:
        return None
    cw, ch = cells[0].size
    grid = Image.new("RGB", (cw * len(cells), ch), color=(255, 255, 255))
    for i, c in enumerate(cells):
        grid.paste(c, (i * cw, 0))
    return grid


# ============================================================
# Main
# ============================================================
def main(args):
    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = sorted(d for d in sweep_root.iterdir()
                         if d.is_dir() and (d / "results.csv").is_file())
    method_dirs = order_methods(method_dirs)
    print(f"[grid] methods in order: {[d.name for d in method_dirs]}")

    # load summary.json for speedup labels (if available)
    summary = {}
    sj = sweep_root / "summary.json"
    if sj.is_file():
        with open(sj) as f:
            summary = json.load(f)

    # pick images
    n_max = sum(1 for sub in (method_dirs[0]).iterdir()
                if sub.is_dir() and sub.name.startswith("img_"))
    if args.indices:
        indices = args.indices
    elif args.pick == "diverse":
        indices = pick_diverse(sweep_root, args.n_grids)
        if indices is None:
            indices = pick_first(args.n_grids, n_max)
    else:
        indices = pick_first(args.n_grids, n_max)
    print(f"[grid] image indices to render: {indices}")

    font = try_font(size=20)

    for idx in indices:
        grid = build_grid(
            sweep_root, method_dirs, idx, summary,
            cell_size=args.cell_size, font=font,
        )
        if grid is None:
            print(f"[grid] skipping idx {idx} (no images)")
            continue
        out_path = out_dir / f"compare_img_{idx:03d}.png"
        grid.save(out_path)
        print(f"[grid] saved {out_path}  ({grid.size[0]}x{grid.size[1]})")

    print(f"\n[grid] done. View the comparison grids in {out_dir}")
    print(f"[grid] Tip: open them locally and compare the inpainted region")
    print(f"           across methods to judge plausibility (not target-fidelity).")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_grids", type=int, default=10,
                   help="Number of comparison grids to produce.")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="Explicit image indices (overrides --pick/--n_grids).")
    p.add_argument("--pick", type=str, default="first",
                   choices=["first", "diverse"],
                   help="'first': first N indices. 'diverse': N images where "
                        "freqspec_default and target_s40 disagree most.")
    p.add_argument("--cell_size", type=int, default=384,
                   help="Side length of each method cell in pixels.")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
