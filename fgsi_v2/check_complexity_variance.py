#!/usr/bin/env python
"""
Quick diagnostic: does Places2 really have low within-dataset
complexity variance, explaining why correlations are null there?

This loads the same data analyze_nfe_complexity_v2.py used and
reports: for each proxy, the within-dataset standard deviation
divided by between-dataset mean. Higher = more variable input
distribution = more room for adaptive NFE allocation to show up.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


# Re-import the same proxy functions (kept here to be standalone).
def _load_gray(path, size=256):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_mask(path, size=256):
    if not path.is_file():
        return None
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)


def proxy_gradient(gray):
    gy = np.diff(gray, axis=0)[:, :-1]
    gx = np.diff(gray, axis=1)[:-1, :]
    return float(np.sqrt(gx * gx + gy * gy).mean())


def proxy_local_entropy(gray, block=16):
    H, W = gray.shape
    H_b = (H // block) * block
    W_b = (W // block) * block
    g = gray[:H_b, :W_b]
    rb = g.reshape(H_b // block, block, W_b // block, block)
    rb = rb.transpose(0, 2, 1, 3).reshape(-1, block * block)
    ents = []
    for blk in rb:
        hist, _ = np.histogram(blk, bins=8, range=(0.0, 1.0))
        p = hist.astype(np.float64) / max(hist.sum(), 1)
        p = p[p > 0]
        ents.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(ents)) if ents else 0.0


def load_freqspec_default_proxies(sweep_root, label, n_max=100):
    """Load and compute proxies for freqspec_default images."""
    sweep_root = Path(sweep_root)
    m_dir = sweep_root / "freqspec_default"
    csv_path = m_dir / "results.csv"
    if not csv_path.is_file():
        return []
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                idx = int(r["idx"])
            except (KeyError, ValueError):
                continue
            if len(out) >= n_max:
                break
            gt_path = m_dir / f"img_{idx:03d}" / "gt.png"
            if not gt_path.is_file():
                continue
            try:
                g = _load_gray(gt_path)
                out.append({
                    "idx": idx,
                    "gradient": proxy_gradient(g),
                    "entropy":  proxy_local_entropy(g),
                })
            except Exception:
                continue
    return out


def main(args):
    print()
    print("=" * 72)
    print("VARIANCE DIAGNOSTIC: how variable is each dataset, per proxy?")
    print("=" * 72)
    print()
    print("If a dataset has LOW within-dataset variance for a proxy,")
    print("there is no room for that proxy to correlate with NFE — even if")
    print("the underlying adaptive mechanism is working perfectly.")
    print()

    all_data = {}
    for label, root in [
        ("FFHQ", args.ffhq_root),
        ("Places2", args.places2_root),
        ("COCO", args.coco_root),
    ]:
        if not root:
            continue
        print(f"[load] {label}...")
        data = load_freqspec_default_proxies(root, label, n_max=args.n_max)
        if data:
            all_data[label] = data
            print(f"   {len(data)} images loaded")

    print()
    print(f"{'Proxy':<12s} {'Dataset':<10s} {'Mean':>10s} {'Std':>10s} "
          f"{'CV':>8s} {'IQR/Mean':>10s}")
    print("-" * 72)
    for proxy in ["gradient", "entropy"]:
        for label, data in all_data.items():
            vals = np.array([d[proxy] for d in data])
            mean = float(vals.mean())
            std = float(vals.std())
            cv = std / mean if mean > 0 else float("nan")
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr_norm = (q75 - q25) / mean if mean > 0 else float("nan")
            print(f"{proxy:<12s} {label:<10s} {mean:>10.4f} {std:>10.4f} "
                  f"{cv:>8.3f} {iqr_norm:>10.3f}")
        print()

    print("-" * 72)
    print("Interpretation:")
    print("  CV (coefficient of variation) measures relative spread.")
    print("  Higher CV  -> more variable inputs -> easier to detect")
    print("                NFE-complexity correlation.")
    print("  Lower CV   -> more uniform inputs -> harder to detect")
    print("                correlation even if real.")
    print()
    print("If Places2 has notably lower CV than COCO on the proxies, that")
    print("supports the explanation that Places2's null correlation is")
    print("due to homogeneous input complexity rather than a method failure.")
    print()


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--places2_root", type=str, default="")
    p.add_argument("--ffhq_root",    type=str, default="")
    p.add_argument("--coco_root",    type=str, default="")
    p.add_argument("--n_max", type=int, default=100,
                   help="Max images per dataset to process (default 100).")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
