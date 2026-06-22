#!/usr/bin/env python
"""
analyze_sr.py — aggregate sr_baseline_sweep.py outputs into the AAAI SR
operating-point table (Table-2 style).

Reads each method's results.csv (time / NFE / accept / PSNR / SSIM / HH-PSNR)
and the saved per-image out.png to compute LPIPS and trajectory-divergence
LPIPSt (LPIPS to the target_s50 output of the same image). Emits:

    sr_table.csv     per-method means (speedup, NFE, accept, PSNR, SSIM,
                     HH-PSNR, LPIPS, LPIPSt)
    sr_table.tex     LaTeX version mirroring the inpainting Table 2 layout

LPIPS requires the `lpips` package; if unavailable, those columns are omitted
and the table still reports PSNR / SSIM / HH-PSNR / speedup / NFE / accept.

Usage:
    python analyze_sr.py --out_root /mnt/HDD_12TB/bam_ki/results/sr_sweep_div2k \
        --ref_method target_s50
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    import torch
    import lpips as lpips_lib
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_rgb(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return a  # HxWx3 in [-1,1]


def main(args):
    methods = [d for d in sorted(os.listdir(args.out_root))
               if os.path.isdir(os.path.join(args.out_root, d))
               and os.path.isfile(os.path.join(args.out_root, d, "results.csv"))]
    if not methods:
        print(f"[analyze-sr] no method dirs under {args.out_root}")
        return
    print(f"[analyze-sr] methods: {methods}")

    # per-method per-image rows keyed by idx
    rows = {m: {int(r["idx"]): r for r in read_csv(
            os.path.join(args.out_root, m, "results.csv"))} for m in methods}

    lpips_fn = None
    if _HAS_LPIPS and not args.no_lpips:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_fn = lpips_lib.LPIPS(net="alex").to(dev).eval()
        print(f"[analyze-sr] LPIPS(alex) on {dev}")
    else:
        print("[analyze-sr] LPIPS unavailable -> PSNR/SSIM/HH only")

    def lpips_dist(a, b):
        if lpips_fn is None:
            return float("nan")
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(next(lpips_fn.parameters()).device)
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(ta.device)
        with torch.no_grad():
            return float(lpips_fn(ta, tb).item())

    ref = args.ref_method
    agg = defaultdict(lambda: defaultdict(list))
    for m in methods:
        for idx, r in rows[m].items():
            for k in ("time_sec", "psnr", "ssim", "hh_psnr"):
                try:
                    agg[m][k].append(float(r[k]))
                except (ValueError, KeyError):
                    pass
            agg[m]["target_nfe"].append(float(r["target_nfe"]))
            agg[m]["draft_nfe"].append(float(r.get("draft_nfe", 0) or 0))
            if r.get("accept_rate", ""):
                agg[m]["accept_rate"].append(float(r["accept_rate"]))
            # LPIPS vs HR and LPIPSt vs ref output
            out_p = os.path.join(args.out_root, m, f"img_{idx:04d}", "out.png")
            hr_p = os.path.join(args.out_root, m, f"img_{idx:04d}", "hr.png")
            ref_out_p = os.path.join(args.out_root, ref, f"img_{idx:04d}", "out.png")
            if lpips_fn is not None and os.path.isfile(out_p):
                out_img = load_rgb(out_p)
                if os.path.isfile(hr_p):
                    agg[m]["lpips"].append(lpips_dist(out_img, load_rgb(hr_p)))
                if os.path.isfile(ref_out_p):
                    agg[m]["lpips_t"].append(lpips_dist(out_img, load_rgb(ref_out_p)))

    ref_time = float(np.mean(agg[ref]["time_sec"])) if agg[ref]["time_sec"] else None

    def mean(m, k):
        v = agg[m].get(k, [])
        return float(np.mean(v)) if v else float("nan")

    cols = ["method", "target_nfe", "accept", "speedup", "psnr", "ssim",
            "hh_psnr", "lpips", "lpips_t"]
    table = []
    for m in methods:
        sp = (ref_time / mean(m, "time_sec")) if (ref_time and mean(m, "time_sec") > 0) else float("nan")
        table.append({
            "method": m,
            "target_nfe": mean(m, "target_nfe"),
            "accept": mean(m, "accept_rate"),
            "speedup": sp,
            "psnr": mean(m, "psnr"), "ssim": mean(m, "ssim"),
            "hh_psnr": mean(m, "hh_psnr"),
            "lpips": mean(m, "lpips"), "lpips_t": mean(m, "lpips_t"),
        })

    csv_out = os.path.join(args.out_root, "sr_table.csv")
    with open(csv_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in table:
            wr.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
                         for k in cols})
    print(f"[analyze-sr] wrote {csv_out}")

    # console + LaTeX
    hdr = f"{'method':18s} {'tgtNFE':>7s} {'acc':>6s} {'spd':>6s} {'PSNR':>7s} {'SSIM':>7s} {'HH':>7s} {'LPIPS':>7s} {'LPIPSt':>7s}"
    print("\n" + hdr); print("-" * len(hdr))
    for row in table:
        print(f"{row['method']:18s} {row['target_nfe']:7.1f} {row['accept']:6.3f} "
              f"{row['speedup']:6.2f} {row['psnr']:7.2f} {row['ssim']:7.4f} "
              f"{row['hh_psnr']:7.2f} {row['lpips']:7.4f} {row['lpips_t']:7.4f}")

    tex = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
           r"Method & Tgt.\ NFE$\downarrow$ & Accept$\uparrow$ & Speedup$\uparrow$ & "
           r"PSNR$\uparrow$ & SSIM$\uparrow$ & HH-PSNR$\uparrow$ & LPIPS$\downarrow$ & LPIPS$_t\downarrow$ \\",
           r"\midrule"]
    for row in table:
        acc = "--" if np.isnan(row["accept"]) else f"{row['accept']:.3f}"
        tex.append(f"{row['method'].replace('_', chr(92)+'_')} & {row['target_nfe']:.1f} & {acc} & "
                   f"{row['speedup']:.2f}$\\times$ & {row['psnr']:.2f} & {row['ssim']:.4f} & "
                   f"{row['hh_psnr']:.2f} & {row['lpips']:.4f} & {row['lpips_t']:.4f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_out = os.path.join(args.out_root, "sr_table.tex")
    with open(tex_out, "w") as f:
        f.write("\n".join(tex))
    print(f"[analyze-sr] wrote {tex_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--ref_method", type=str, default="target_s50",
                    help="per-image reference for LPIPSt (trajectory divergence)")
    ap.add_argument("--no_lpips", action="store_true")
    main(ap.parse_args())
