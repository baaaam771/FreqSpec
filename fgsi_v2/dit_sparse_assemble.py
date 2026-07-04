#!/usr/bin/env python
"""
dit_sparse_assemble.py — collect sparse_summary.json files into the roadmap
tables (sparse execution table + suffix/ratio ablation).

Walks --root recursively, reads every sparse_summary.json, and writes
sparse_results.csv plus a markdown table sorted by (selector, suffix_mode,
split, hard_ratio).

Usage:
    python dit_sparse_assemble.py --root /mnt/HDD_12TB/bam_ki/results/dit_in64/sparse
"""
import argparse
import csv
import glob
import json
import os

COLS = ["selector", "suffix_mode", "split", "hard_ratio",
        "per_step_vs_dense", "total_gmac", "fid",
        "wall_per_image_s", "draft_ms", "select_ms", "sparse_ms", "dense_ms"]


def main(args):
    rows = []
    for p in sorted(glob.glob(os.path.join(args.root, "**",
                                           "sparse_summary.json"),
                              recursive=True)):
        with open(p) as f:
            s = json.load(f)
        ms = s.get("per_image_ms", {})
        rows.append(dict(
            selector=s["selector"], suffix_mode=s["suffix_mode"],
            split=s["split"], hard_ratio=s["hard_ratio"],
            per_step_vs_dense=s["flops"]["per_step_vs_dense"],
            total_gmac=s["flops"]["total_gmac"],
            fid=s.get("fid", ""),
            wall_per_image_s=s.get("wall_per_image_s", ""),
            draft_ms=ms.get("draft", ""), select_ms=ms.get("select", ""),
            sparse_ms=ms.get("target_sparse", ""),
            dense_ms=ms.get("target_dense", ms.get("oracle_dense_target", "")),
            path=os.path.dirname(p)))
    rows.sort(key=lambda r: (str(r["selector"]), str(r["suffix_mode"]),
                             r["split"], r["hard_ratio"]))

    out_csv = os.path.join(args.root, "sparse_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + ["path"])
        w.writeheader()
        w.writerows(rows)

    out_md = os.path.join(args.root, "sparse_results.md")
    with open(out_md, "w") as f:
        f.write("| " + " | ".join(COLS) + " |\n")
        f.write("|" + "---|" * len(COLS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in COLS) + " |\n")
    print(f"[assemble] {len(rows)} runs -> {out_csv}, {out_md}")
    for r in rows:
        print(f"  {r['selector']:9s} {r['suffix_mode']:11s} "
              f"m={r['split']} r={r['hard_ratio']} "
              f"mac={r['per_step_vs_dense']} fid={r['fid']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    main(ap.parse_args())
