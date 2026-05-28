#!/usr/bin/env python
"""
analyze_speed_matched.py — Compute region-aware metrics and build the
speed-matched comparison table from baseline_sweep.py outputs.

Workflow:
  1. Read each method's results.csv (timing, NFE) from the sweep output.
  2. For every (method, image), load gt.png / out.png / mask.png and compute
     PSNR, SSIM, LPIPS, masked LPIPS, boundary LPIPS via metrics_extended.
  3. Aggregate per-method means + std.
  4. Compute speedup = target_s50_time / method_time (per image, then mean).
  5. Find speed-matched pairs: for each FreqSpec preset, find the reduced-step
     target with the closest mean speedup, and print them side by side.
  6. Emit a LaTeX table (BMVC style) ready to paste.

Example:
    python analyze_speed_matched.py \\
        --sweep_root /mnt/HDD_12TB/bam_ki/results/baseline_sweep_places2 \\
        --dataset_name Places2 \\
        --boundary_k 8
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from metrics_extended import RegionMetrics


def load_png(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask_png(path):
    m = np.array(Image.open(path).convert("L"))
    return (m > 127).astype(np.float32)


def read_method_csv(method_dir):
    """Return list of row dicts from a method's results.csv."""
    csv_path = os.path.join(method_dir, "results.csv")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main(args):
    sweep_root = Path(args.sweep_root)
    method_dirs = sorted(
        d for d in sweep_root.iterdir()
        if d.is_dir() and os.path.isfile(d / "results.csv")
    )
    print(f"[analyze] found {len(method_dirs)} methods: "
          f"{[d.name for d in method_dirs]}")

    metric_engine = RegionMetrics(device=args.device, boundary_k=args.boundary_k)

    # per-method: lists of metrics + timing
    agg = defaultdict(lambda: defaultdict(list))

    for mdir in method_dirs:
        method = mdir.name
        rows = read_method_csv(mdir)
        for r in rows:
            idx = int(r["idx"])
            img_dir = mdir / f"img_{idx:03d}"
            gt_p = img_dir / "gt.png"
            out_p = img_dir / "out.png"
            mask_p = img_dir / "mask.png"
            if not (gt_p.exists() and out_p.exists() and mask_p.exists()):
                continue
            gt = load_png(gt_p)
            out = load_png(out_p)
            mask = load_mask_png(mask_p)
            m = metric_engine.compute(out, gt, mask)

            for k, v in m.items():
                agg[method][k].append(v)
            agg[method]["time_sec"].append(float(r["time_sec"]))
            agg[method]["target_nfe"].append(float(r["target_nfe"]))
            if r["draft_nfe"]:
                agg[method]["draft_nfe"].append(float(r["draft_nfe"]))
            if r["accept_rate"]:
                agg[method]["accept_rate"].append(float(r["accept_rate"]))

    # reference time = target_s50 mean time (for speedup)
    ref_key = None
    for cand in ["target_s50", "target_s50".replace("50", str(args.ref_steps))]:
        if cand in agg:
            ref_key = cand
            break
    if ref_key is None:
        # fall back: slowest method = reference
        ref_key = max(agg, key=lambda k: np.mean(agg[k]["time_sec"]))
    ref_time = np.mean(agg[ref_key]["time_sec"])
    print(f"[analyze] reference = {ref_key}, mean time = {ref_time:.3f}s")

    # build summary
    summary = {}
    for method, d in agg.items():
        n = len(d["lpips"])
        speedup = ref_time / np.mean(d["time_sec"])
        summary[method] = {
            "n": n,
            "speedup": speedup,
            "time": np.mean(d["time_sec"]),
            "psnr": np.mean(d["psnr"]),
            "ssim": np.mean(d["ssim"]),
            "lpips": np.mean(d["lpips"]),
            "lpips_std": np.std(d["lpips"]),
            "masked_lpips": np.mean(d["masked_lpips"]),
            "boundary_lpips": np.mean(d["boundary_lpips"]),
            "masked_psnr": np.mean(d["masked_psnr"]),
            "accept_rate": np.mean(d["accept_rate"]) if d["accept_rate"] else None,
        }

    # ---- print full summary ----
    print(f"\n{'='*100}")
    print(f"FULL SUMMARY — {args.dataset_name} (n per method shown)")
    print(f"{'='*100}")
    hdr = (f"{'method':22} {'n':>4} {'speedup':>8} {'time':>7} "
           f"{'PSNR':>6} {'SSIM':>6} {'LPIPS':>7} {'mLPIPS':>7} "
           f"{'bLPIPS':>7} {'accept':>7}")
    print(hdr)
    print("-" * len(hdr))
    for method in sorted(summary, key=lambda k: -summary[k]["speedup"]):
        s = summary[method]
        acc = f"{s['accept_rate']:.3f}" if s["accept_rate"] is not None else "  -  "
        print(f"{method:22} {s['n']:>4} {s['speedup']:>7.2f}x {s['time']:>6.2f}s "
              f"{s['psnr']:>6.2f} {s['ssim']:>6.3f} {s['lpips']:>7.4f} "
              f"{s['masked_lpips']:>7.4f} {s['boundary_lpips']:>7.4f} {acc:>7}")

    # ---- speed-matched pairs ----
    print(f"\n{'='*100}")
    print("SPEED-MATCHED COMPARISON (FreqSpec vs closest reduced-step target)")
    print(f"{'='*100}")
    fs_methods = [m for m in summary if m.startswith("freqspec")]
    tgt_methods = [m for m in summary if m.startswith("target")]

    pairs = []
    for fs in fs_methods:
        fs_sp = summary[fs]["speedup"]
        # closest target by speedup
        closest = min(tgt_methods,
                      key=lambda t: abs(summary[t]["speedup"] - fs_sp))
        pairs.append((fs, closest))
        sf, st = summary[fs], summary[closest]
        print(f"\n  {fs} (speedup {sf['speedup']:.2f}x) "
              f"vs {closest} (speedup {st['speedup']:.2f}x):")
        print(f"    {'metric':16} {'FreqSpec':>10} {'Target':>10} {'winner':>8}")
        for k, lower_better in [("lpips", True), ("masked_lpips", True),
                                ("boundary_lpips", True), ("psnr", False),
                                ("ssim", False)]:
            fv, tv = sf[k], st[k]
            if lower_better:
                win = "FreqSpec" if fv < tv else "Target"
            else:
                win = "FreqSpec" if fv > tv else "Target"
            print(f"    {k:16} {fv:>10.4f} {tv:>10.4f} {win:>8}")

    # ---- LaTeX table ----
    latex = build_latex_table(summary, pairs, args.dataset_name)
    out_tex = os.path.join(args.sweep_root, "speed_matched_table.tex")
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"\n[analyze] LaTeX table -> {out_tex}")

    # ---- save summary json ----
    out_json = os.path.join(args.sweep_root, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[analyze] summary -> {out_json}")


def build_latex_table(summary, pairs, dataset_name):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Speed-matched comparison on " + dataset_name +
        r". FreqSpec (full 50-step schedule) vs.\ reduced-step target "
        r"baselines at comparable speedup. Boundary LPIPS is computed in a "
        r"narrow band around the mask edge. Lower LPIPS is better.}",
        r"\label{tab:speed_matched_" + dataset_name.lower() + "}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Method & Steps & Speedup$\uparrow$ & PSNR$\uparrow$ & "
        r"LPIPS$\downarrow$ & mLPIPS$\downarrow$ & bLPIPS$\downarrow$ \\",
        r"\midrule",
    ]
    # one block per speed-matched pair: target then freqspec
    seen = set()
    for fs, tgt in pairs:
        for method in [tgt, fs]:
            if method in seen:
                continue
            seen.add(method)
            s = summary[method]
            steps = "50" if method.startswith("freqspec") else \
                method.replace("target_s", "")
            name = method.replace("_", r"\_")
            bold = method.startswith("freqspec")
            fmt = (lambda x, d=4: (r"\textbf{" + f"{x:.{d}f}" + "}") if bold
                   else f"{x:.{d}f}")
            lines.append(
                f"{name} & {steps} & {s['speedup']:.2f}$\\times$ & "
                f"{fmt(s['psnr'],2)} & {fmt(s['lpips'])} & "
                f"{fmt(s['masked_lpips'])} & {fmt(s['boundary_lpips'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="Places2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--boundary_k", type=int, default=8,
                   help="Half-width (pixels) of the boundary band.")
    p.add_argument("--ref_steps", type=int, default=50,
                   help="Target step count used as 1.0x speedup reference.")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
