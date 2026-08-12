#!/usr/bin/env python
"""
analyze_difficulty_strata.py — Does the FreqSpec-vs-reduced-step gap change
with input difficulty? (GPU-free; reads phase2_per_image.csv)

Pre-defined strata (no selection bias): quartiles of mask_coverage and of
mask_complexity, computed per run over the shared 100-image manifest.
For each stratum, prints mean gt_masked_lpips and mean lpips_t per method,
plus the freqspec_default minus matched-speed-target delta.

If the gap shrinks or flips in the hardest quartile, that is the regime
argument; if it stays flat, hard-workload hopes rest entirely on the
large-mask / instance-mask sweeps.

Usage:
    python analyze_difficulty_strata.py --csv phase2_per_image.csv \\
        --matched target_s44 --fs freqspec_default
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--fs", default="freqspec_default")
    p.add_argument("--matched", default="target_s44",
                   help="Reduced-step method whose speed matches --fs "
                        "(pick per Phase-2 speedups; s44 ~ default on "
                        "COCO/Places2, s40 ~ default on FFHQ).")
    p.add_argument("--by", default="mask_coverage",
                   choices=["mask_coverage", "mask_complexity"])
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    runs = sorted({r["run"] for r in rows})

    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        # stratum boundaries from the shared manifest (any single method)
        base = [r for r in rr if r["method"] == args.fs]
        vals = {int(r["idx"]): float(r[args.by]) for r in base
                if r[args.by] not in ("", "nan")}
        if not vals:
            print(f"[strata] {run}: no {args.by} values — skip")
            continue
        qs = np.percentile(list(vals.values()), [25, 50, 75])

        def stratum(idx):
            v = vals.get(idx)
            if v is None:
                return None
            return int(np.searchsorted(qs, v, side="right"))  # 0..3

        print(f"\n===== {run}  (strata by {args.by}; Q edges "
              f"{[round(float(q),3) for q in qs]}) =====")
        print(f"{'Q':>2} {'n':>4} | {'fs gt_m':>8} {'tgt gt_m':>8} {'Δgt_m':>8} "
              f"| {'fs lp_t':>8} {'tgt lp_t':>8}")
        for q in range(4):
            idxs = [i for i in vals if stratum(i) == q]
            def mmean(method, col):
                xs = [float(r[col]) for r in rr
                      if r["method"] == method and int(r["idx"]) in idxs
                      and r[col] not in ("", "nan")]
                return float(np.mean(xs)) if xs else float("nan")
            fg = mmean(args.fs, "gt_masked_lpips")
            tg = mmean(args.matched, "gt_masked_lpips")
            fl = mmean(args.fs, "lpips_t")
            tl = mmean(args.matched, "lpips_t")
            print(f"{q:>2} {len(idxs):>4} | {fg:8.4f} {tg:8.4f} {fg-tg:+8.4f} "
                  f"| {fl:8.4f} {tl:8.4f}")
        print("  (Δgt_m < 0 in Q3 would mean FreqSpec wins on the hardest "
              "quartile; Δ shrinking with Q = regime trend)")


if __name__ == "__main__":
    main()
