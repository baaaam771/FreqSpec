#!/usr/bin/env python
"""
imagenet_parquet_to_folder.py -- convert HF ImageNet-64 parquet shards into a
class-conditional ImageFolder layout (train/<label>/<idx>.png) that the
`--dataset imagenet` loader expects.

Handles the common HF schema where each row has an image column holding either
raw bytes, a dict {"bytes": ..., "path": ...}, or a PIL-decodable value, plus an
integer label column. Column names are auto-detected (override with flags).

Usage:
    python imagenet_parquet_to_folder.py \
        --parquet_glob "/mnt/HDD_12TB/bam_ki/datasets/imagenet64/data/*.parquet" \
        --out_root /mnt/HDD_12TB/bam_ki/datasets/imagenet64/train \
        --img_size 64
Run once for train shards and once for validation shards (different out_root).
"""
import argparse
import glob
import io
import os

from PIL import Image


def _find_col(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def _to_pil(val):
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, dict):
        if val.get("bytes"):
            return Image.open(io.BytesIO(val["bytes"]))
        if val.get("path"):
            return Image.open(val["path"])
        raise ValueError(f"image dict has no bytes/path: {list(val.keys())}")
    if isinstance(val, (bytes, bytearray)):
        return Image.open(io.BytesIO(val))
    if isinstance(val, str):
        return Image.open(val)
    raise ValueError(f"unrecognized image value type: {type(val)}")


def main(a):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(a.parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet at {a.parquet_glob}")
    print(f"[conv] {len(files)} parquet shards")
    os.makedirs(a.out_root, exist_ok=True)

    n = 0
    img_col = a.image_col
    lbl_col = a.label_col
    for fi, f in enumerate(files):
        t = pq.read_table(f)
        if img_col is None:
            img_col = _find_col(t.column_names, ["image", "img", "jpg", "png"])
        if lbl_col is None:
            lbl_col = _find_col(t.column_names, ["label", "labels", "class", "y"])
        if img_col is None or lbl_col is None:
            raise ValueError(f"could not find image/label cols in {t.column_names}; "
                             f"pass --image_col/--label_col")
        rows = t.to_pylist()
        for r in rows:
            lbl = int(r[lbl_col])
            d = os.path.join(a.out_root, f"{lbl:04d}")
            os.makedirs(d, exist_ok=True)
            try:
                im = _to_pil(r[img_col]).convert("RGB")
            except Exception as e:
                print(f"  skip row (decode fail): {e}")
                continue
            if a.img_size and im.size != (a.img_size, a.img_size):
                im = im.resize((a.img_size, a.img_size), Image.BICUBIC)
            im.save(os.path.join(d, f"{n:08d}.png"))
            n += 1
            if a.limit and n >= a.limit:
                print(f"[conv] hit limit {a.limit}"); _summary(a, n); return
        print(f"[conv] shard {fi+1}/{len(files)} done, total {n} images")
    _summary(a, n)


def _summary(a, n):
    classes = [d for d in os.listdir(a.out_root)
               if os.path.isdir(os.path.join(a.out_root, d))]
    print(f"[conv] wrote {n} images into {len(classes)} class folders under {a.out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_glob", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=64, help="resize to NxN (0=keep)")
    p.add_argument("--image_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--limit", type=int, default=0, help="stop after N images (debug)")
    main(p.parse_args())
