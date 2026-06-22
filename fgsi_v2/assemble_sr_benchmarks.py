#!/usr/bin/env python
"""
assemble_sr_benchmarks.py — combine per-dataset sr_table.csv into one
cross-dataset SR table for the AAAI paper.

Reads <out_root>/<DS>/sr_table.csv for each dataset and emits a compact table
focused on the operating-point comparison the reviewer cares about:
target_s50 (reference), target_s30 (strong reduced-step baseline), and
freqspec_default (the verifier operating point), per dataset.

Outputs:
    <out_root>/sr_benchmarks.csv
    <out_root>/sr_benchmarks.tex
"""
import argparse
import csv
import os


FOCUS = ["target_s50", "target_s30", "freqspec_default"]
METRICS = ["target_nfe", "accept", "speedup", "psnr", "ssim", "hh_psnr", "lpips", "lpips_t"]


def read_table(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["method"]] = r
    return rows


def fmt(v):
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return "--"


def main(args):
    combined = []  # (dataset, method, row)
    for ds in args.datasets:
        p = os.path.join(args.out_root, ds, "sr_table.csv")
        if not os.path.isfile(p):
            print(f"[assemble] missing {p}, skipping {ds}")
            continue
        tbl = read_table(p)
        for m in FOCUS:
            if m in tbl:
                combined.append((ds, m, tbl[m]))

    # CSV
    csv_out = os.path.join(args.out_root, "sr_benchmarks.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method"] + METRICS)
        for ds, m, r in combined:
            w.writerow([ds, m] + [r.get(k, "") for k in METRICS])
    print(f"[assemble] wrote {csv_out}")

    # console
    hdr = f"{'dataset':9s} {'method':17s} {'tgtNFE':>7s} {'acc':>6s} {'spd':>6s} {'PSNR':>7s} {'SSIM':>7s} {'HH':>7s} {'LPIPS':>7s} {'LPIPSt':>7s}"
    print("\n" + hdr); print("-" * len(hdr))
    last = None
    for ds, m, r in combined:
        tag = ds if ds != last else ""
        last = ds
        print(f"{tag:9s} {m:17s} {fmt(r.get('target_nfe')):>7s} {fmt(r.get('accept')):>6s} "
              f"{fmt(r.get('speedup')):>6s} {fmt(r.get('psnr')):>7s} {fmt(r.get('ssim')):>7s} "
              f"{fmt(r.get('hh_psnr')):>7s} {fmt(r.get('lpips')):>7s} {fmt(r.get('lpips_t')):>7s}")

    # LaTeX
    tex = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
           r"Dataset & Method & Tgt.\ NFE & Accept & Speedup & PSNR & SSIM & HH-PSNR & LPIPS & LPIPS$_t$ \\",
           r"\midrule"]
    last = None
    for ds, m, r in combined:
        tag = ds if ds != last else ""
        if ds != last and last is not None:
            tex.append(r"\midrule")
        last = ds
        acc = fmt(r.get("accept"))
        tex.append(f"{tag} & {m.replace('_', chr(92)+'_')} & {fmt(r.get('target_nfe'))} & {acc} & "
                   f"{fmt(r.get('speedup'))}$\\times$ & {fmt(r.get('psnr'))} & {fmt(r.get('ssim'))} & "
                   f"{fmt(r.get('hh_psnr'))} & {fmt(r.get('lpips'))} & {fmt(r.get('lpips_t'))} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_out = os.path.join(args.out_root, "sr_benchmarks.tex")
    with open(tex_out, "w") as f:
        f.write("\n".join(tex))
    print(f"\n[assemble] wrote {tex_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--datasets", type=str, nargs="+",
                    default=["Set5", "Set14", "BSD100", "Urban100"])
    main(ap.parse_args())
