#!/usr/bin/env python
"""
preprocess_imagenet_ref.py — build 256/512 FID reference folders from the
downloaded ImageNet-1k validation split, using the official DiT/ADM
center-crop-then-resize preprocessing so FID matches published numbers.

Handles both layouts the HF download may produce:
  (a) parquet shards under raw_dir/data/  (image bytes in an 'image' column)
  (b) a folder of image files (any nesting)

Center-crop: crop the largest centered square (short side), then bicubic
resize to the target. Saved as PNG (lossless -> no JPEG artifacts in the FID
reference).

Usage:
    python preprocess_imagenet_ref.py --raw_dir .../imagenet_val_raw \
        --out256 .../imagenet256_val --out512 .../imagenet512_val --num 50000
"""
import argparse
import glob
import io
import os

from PIL import Image


def center_crop_resize(img, size):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.BICUBIC)


def iter_parquet(raw_dir):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.parquet"),
                             recursive=True))
    for pf in files:
        t = pq.read_table(pf)
        col = "image" if "image" in t.column_names else t.column_names[0]
        for rec in t.column(col):
            d = rec.as_py()
            if isinstance(d, dict) and "bytes" in d:
                yield Image.open(io.BytesIO(d["bytes"]))
            elif isinstance(d, (bytes, bytearray)):
                yield Image.open(io.BytesIO(d))


def iter_images(raw_dir):
    exts = ("*.JPEG", "*.jpeg", "*.jpg", "*.png")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(raw_dir, "**", e), recursive=True)
    for fp in sorted(files):
        try:
            yield Image.open(fp)
        except Exception:
            continue


def main(args):
    os.makedirs(args.out256, exist_ok=True)
    os.makedirs(args.out512, exist_ok=True)
    has_parquet = bool(glob.glob(os.path.join(args.raw_dir, "**", "*.parquet"),
                                 recursive=True))
    src = iter_parquet(args.raw_dir) if has_parquet else iter_images(args.raw_dir)
    print(f"[prep] source: {'parquet' if has_parquet else 'image files'}")

    n = 0
    for img in src:
        if n >= args.num:
            break
        try:
            center_crop_resize(img, 256).save(
                os.path.join(args.out256, f"{n:06d}.png"))
            center_crop_resize(img, 512).save(
                os.path.join(args.out512, f"{n:06d}.png"))
        except Exception as e:
            print(f"[prep] skip {n}: {e}")
            continue
        n += 1
        if n % 2000 == 0:
            print(f"[prep] {n}/{args.num}")
    print(f"[prep] wrote {n} images to {args.out256} and {args.out512}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out256", type=str, required=True)
    ap.add_argument("--out512", type=str, required=True)
    ap.add_argument("--num", type=int, default=50000)
    main(ap.parse_args())
