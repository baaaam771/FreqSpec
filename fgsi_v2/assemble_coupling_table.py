#!/usr/bin/env python
"""
assemble_coupling_table.py — SR coupling x tolerance operating-point table.

Reads per-coupling sr_table.csv files (each produced by sr_baseline_sweep.py +
analyze_sr.py at one saliency_x0_coupling value) and assembles the coupling x
tolerance grid the AAAI SR subsection needs: for each (coupling, tolerance) the
accept rate, target NFE, wall-clock speedup, PSNR/SSIM/HH-PSNR/LPIPS/LPIPSt, plus
the deployed Full-FreqSpec AURC for that coupling (parsed from the matching
reliability summary.txt if provided).

Usage:
    python assemble_coupling_table.py \
        --couplings 0.0 0.3 0.6 0.9 \
        --sweep_dirs results/sr_cpl_op_0.0 results/sr_cpl_op_0.3 \
                     results/sr_cpl_op_0.6 results/sr_cpl_op_0.9 \
        --aurc_summaries results/sr_cpl_rel_0.0/analysis/summary.txt ... (optional) \
        --out_dir results/sr_coupling_table
"""
import argparse
import csv
import os
import re

TOL = ["freqspec_strict", "freqspec_mid", "freqspec_default"]
METRICS = ["accept", "target_nfe", "speedup", "psnr", "ssim", "hh_psnr",
           "lpips", "lpips_t"]


def read_table(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["method"]] = r
    return out


def parse_aurc(summary_path):
    if not summary_path or not os.path.isfile(summary_path):
        return None
    with open(summary_path) as f:
        for line in f:
            m = re.search(r"Full FreqSpec\s+([0-9]+\.[0-9]+)", line)
            if m:
                return float(m.group(1))
    return None


def fmt(v):
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return "--"


def main(args):
    aurc_by_c = {}
    if args.aurc_summaries:
        for c, s in zip(args.couplings, args.aurc_summaries):
            aurc_by_c[c] = parse_aurc(s)

    combined = []  # (coupling, tol, row, aurc)
    for c, d in zip(args.couplings, args.sweep_dirs):
        tbl = read_table(os.path.join(d, "sr_table.csv"))
        for t in TOL:
            if t in tbl:
                combined.append((c, t, tbl[t], aurc_by_c.get(c)))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_out = os.path.join(args.out_dir, "coupling_table.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coupling", "tolerance"] + METRICS + ["full_aurc"])
        for c, t, r, a in combined:
            w.writerow([c, t.replace("freqspec_", "")]
                       + [r.get(k, "") for k in METRICS]
                       + [f"{a:.5f}" if a is not None else ""])
    print(f"[assemble] wrote {csv_out}")

    hdr = (f"{'cpl':>4s} {'tol':8s} {'acc':>6s} {'tgtNFE':>7s} {'spd':>6s} "
           f"{'PSNR':>7s} {'SSIM':>7s} {'HH':>7s} {'LPIPS':>7s} {'LPIPSt':>7s} {'AURC':>8s}")
    print("\n" + hdr); print("-" * len(hdr))
    last = None
    for c, t, r, a in combined:
        cc = f"{c}" if c != last else ""
        last = c
        print(f"{cc:>4s} {t.replace('freqspec_',''):8s} {fmt(r.get('accept')):>6s} "
              f"{fmt(r.get('target_nfe')):>7s} {fmt(r.get('speedup')):>6s} "
              f"{fmt(r.get('psnr')):>7s} {fmt(r.get('ssim')):>7s} {fmt(r.get('hh_psnr')):>7s} "
              f"{fmt(r.get('lpips')):>7s} {fmt(r.get('lpips_t')):>7s} "
              f"{(f'{a:.5f}' if a is not None else '--'):>8s}")

    # LaTeX
    tex = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r"Coupling & Tol. & Accept & Tgt.\ NFE & Speedup & PSNR & SSIM & "
           r"HH-PSNR & LPIPS & LPIPS$_t$ & AURC \\", r"\midrule"]
    last = None
    for c, t, r, a in combined:
        if c != last and last is not None:
            tex.append(r"\midrule")
        cc = f"{c}" if c != last else ""
        last = c
        tex.append(f"{cc} & {t.replace('freqspec_','')} & {fmt(r.get('accept'))} & "
                   f"{fmt(r.get('target_nfe'))} & {fmt(r.get('speedup'))}$\\times$ & "
                   f"{fmt(r.get('psnr'))} & {fmt(r.get('ssim'))} & {fmt(r.get('hh_psnr'))} & "
                   f"{fmt(r.get('lpips'))} & {fmt(r.get('lpips_t'))} & "
                   f"{(f'{a:.5f}' if a is not None else '--')} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out_dir, "coupling_table.tex"), "w") as f:
        f.write("\n".join(tex))
    print(f"[assemble] wrote {os.path.join(args.out_dir, 'coupling_table.tex')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--couplings", type=float, nargs="+", required=True)
    ap.add_argument("--sweep_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--aurc_summaries", type=str, nargs="*", default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    main(ap.parse_args())
