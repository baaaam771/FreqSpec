#!/usr/bin/env python
"""
make_master_comparison.py — Build a single overview figure combining
the most informative comparison grids from all three datasets.

Output: one tall PNG showing Places2 / FFHQ / COCO side by side, with
the same set of methods displayed for each dataset's best diverse-pick
image. Useful as a single-figure summary.

Layout:
    Row 1 (Places2):  [GT|Mask|t_s50|t_s40|t_s30|fs_strict|fs_mid|fs_default]
    Row 2 (FFHQ):     [same]
    Row 3 (COCO):     [same]

Usage:
    python make_master_comparison.py \\
        --places2_root /mnt/HDD_12TB/bam_ki/results/sweep_v2_places2_n100 \\
        --ffhq_root    /mnt/HDD_12TB/bam_ki/results/sweep_v2_ffhq_n100 \\
        --coco_root    /mnt/HDD_12TB/bam_ki/results/sweep_v2_coco_n100 \\
        --out          /mnt/HDD_12TB/bam_ki/results/master_summary.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Order in which methods appear (only those present are shown).
METHOD_ORDER = [
    "target_s50", "target_s40", "target_s30",
    "freqspec_strict", "freqspec_mid", "freqspec_default",
]


def try_font(size=18, bold=True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_resize(path, size):
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def pick_diverse_index(sweep_root, fs_method="freqspec_default",
                       ref_method="target_s40"):
    """Pick the single image where fs_method and ref_method differ most
    (highest LPIPS between them). This is the most informative case."""
    import torch, lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="alex").to(device).eval()

    def to_t(p):
        a = np.array(Image.open(p).convert("RGB")
                     .resize((512, 512), Image.BILINEAR))
        t = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    fs_dir = sweep_root / fs_method
    ref_dir = sweep_root / ref_method
    best_score, best_idx = -1.0, 0
    for sub in sorted(fs_dir.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("img_")):
            continue
        fs_p = sub / "out.png"
        ref_p = ref_dir / sub.name / "out.png"
        if not (fs_p.exists() and ref_p.exists()):
            continue
        with torch.no_grad():
            s = float(lp(to_t(fs_p), to_t(ref_p)).item())
        if s > best_score:
            best_score = s
            best_idx = int(sub.name.split("_")[1])
    return best_idx, best_score


def build_row(sweep_root, dataset_label, idx, cell, font_cell, font_dataset):
    """Build one horizontal row for a dataset. Returns PIL.Image."""
    sweep_root = Path(sweep_root)

    # method dirs in preferred order
    available = [d.name for d in sweep_root.iterdir()
                 if d.is_dir() and (d / "results.csv").is_file()]
    method_dirs = [sweep_root / m for m in METHOD_ORDER if m in available]
    if not method_dirs:
        return None

    # speedup info if available
    summary = {}
    sj = sweep_root / "summary.json"
    if sj.is_file():
        with open(sj) as f:
            summary = json.load(f)

    # collect cells: dataset_label, GT, Mask, then methods
    cells = []
    label_h = font_cell.size + 6
    dataset_label_w = cell // 2

    # left-most dataset label strip (vertical text would be nicer, but
    # rendering vertical text reliably is annoying — keep it as a colored bar
    # with horizontal text)
    label_img = Image.new("RGB", (dataset_label_w, cell + label_h),
                          color=(40, 40, 50))
    draw = ImageDraw.Draw(label_img)
    # draw dataset name big & centered
    w_text = draw.textlength(dataset_label, font=font_dataset)
    draw.text(((dataset_label_w - w_text) // 2,
               (cell + label_h - font_dataset.size) // 2),
              dataset_label, fill=(255, 255, 255), font=font_dataset)
    cells.append(label_img)

    # GT, Mask
    sample = method_dirs[0] / f"img_{idx:03d}"
    if (sample / "gt.png").is_file():
        img = load_resize(sample / "gt.png", cell)
        cells.append(_with_label(img, "Ground Truth", font_cell, label_h))
    if (sample / "mask.png").is_file():
        m = Image.open(sample / "mask.png").convert("L").resize(
            (cell, cell), Image.NEAREST).convert("RGB")
        cells.append(_with_label(m, "Mask", font_cell, label_h))

    # methods
    for mdir in method_dirs:
        out_p = mdir / f"img_{idx:03d}" / "out.png"
        if not out_p.is_file():
            continue
        img = load_resize(out_p, cell)
        sp = summary.get(mdir.name, {}).get("speedup")
        label = f"{mdir.name}" + (f"  {sp:.2f}x" if sp is not None else "")
        cells.append(_with_label(img, label, font_cell, label_h))

    # tile
    row = Image.new("RGB", (sum(c.size[0] for c in cells), cells[0].size[1]),
                    color=(255, 255, 255))
    x = 0
    for c in cells:
        row.paste(c, (x, 0))
        x += c.size[0]
    return row


def _with_label(img, text, font, label_h):
    w, h = img.size
    out = Image.new("RGB", (w, h + label_h), color=(245, 245, 245))
    out.paste(img, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((4, 3), text, fill=(15, 15, 15), font=font)
    return out


def main(args):
    cell = args.cell_size
    font_cell = try_font(size=16, bold=True)
    font_dataset = try_font(size=28, bold=True)

    datasets = [
        ("Places2", args.places2_root),
        ("FFHQ", args.ffhq_root),
        ("COCO", args.coco_root),
    ]
    rows = []
    for label, root in datasets:
        if not root:
            continue
        if args.indices and label.lower() in args.indices:
            idx = int(args.indices[label.lower()])
            score = None
        else:
            print(f"[master] picking diverse index for {label}...")
            idx, score = pick_diverse_index(Path(root))
            print(f"[master]   -> img_{idx:03d} (LPIPS diff={score:.4f})")

        row = build_row(root, label, idx, cell, font_cell, font_dataset)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[master] no rows produced; check inputs.")
        return

    # combine vertically
    W = max(r.size[0] for r in rows)
    H = sum(r.size[1] for r in rows)
    final = Image.new("RGB", (W, H), color=(255, 255, 255))
    y = 0
    for r in rows:
        final.paste(r, (0, y))
        y += r.size[1]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path)
    print(f"[master] saved -> {out_path}  ({final.size[0]}x{final.size[1]})")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--places2_root", type=str, default="")
    p.add_argument("--ffhq_root", type=str, default="")
    p.add_argument("--coco_root", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--cell_size", type=int, default=320)
    p.add_argument("--indices", type=str, nargs="*", default=None,
                   help="Optional explicit idx per dataset, "
                        "e.g. --indices places2 34 coco 27")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    # parse --indices key value pairs
    if args.indices:
        if len(args.indices) % 2 != 0:
            raise SystemExit("--indices needs key/value pairs")
        d = {}
        for k, v in zip(args.indices[::2], args.indices[1::2]):
            d[k.lower()] = v
        args.indices = d
    main(args)
