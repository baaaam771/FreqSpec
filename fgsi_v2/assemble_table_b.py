#!/usr/bin/env python
"""
assemble_table_b.py  —  build Table B (saliency-signal ablation).

Merges three sources into one publication table:
  1. saliency_ablation_sweep.py outputs   -> Target NFE, Coverage, time/speedup,
     and per-image out.png for LPIPS_t.
  2. run_target_reference.py outputs       -> target_s50 reference for LPIPS_t.
  3. analyze_verifier_reliability.py output -> Full FreqSpec FAR@50% and AURC
     per saliency config (verifier reliability under that signal).

Mask LPIPS_t / Boundary LPIPS_t are computed with the SAME RegionMetrics engine
the rest of the paper uses (metrics_extended.RegionMetrics.compute_vs_target),
so numbers are consistent with Tables 2-6.

Table B columns (per the experiment design):
  Saliency | Target NFE | Speedup | Coverage | Mask LPIPS_t | Boundary LPIPS_t
           | False-Accept(@50%) | AURC

Outputs:
  <out>/table_b.csv
  <out>/table_b.tex

Example:
  python assemble_table_b.py \\
      --sweep_root /mnt/HDD_12TB/bam_ki/results/saliency_ablation_coco \\
      --ref_dir    /mnt/HDD_12TB/bam_ki/results/saliency_ablation_coco/target_s50 \\
      --verif_root /mnt/HDD_12TB/bam_ki/results/saliency_ablation_coco/verif \\
      --out        /mnt/HDD_12TB/bam_ki/results/saliency_ablation_coco \\
      --boundary_k 8 --far_coverage 0.5
"""
import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image

# canonical row order + display labels
CONFIG_ORDER = [
    ("none_uniform",     "None (uniform tol.)"),
    ("random",           "Random map"),
    ("sobel",            "Sobel gradient"),
    ("variance",         "Local latent variance"),
    ("laplacian",        "Laplacian energy"),
    ("wavelet_only",     "Wavelet only ($A_{\\mathrm{wav}}$)"),
    ("boundary_only",    "Boundary only ($B$)"),
    ("wavelet_boundary", "Wavelet + boundary"),
    ("full",             "Wavelet + boundary + interior"),
]


def _load_png(path):
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path):
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.float32)


def read_runs_csv(sweep_root, config):
    """Return mean target_nfe, mean accept_rate (coverage), mean time."""
    p = os.path.join(sweep_root, f"{config}_runs.csv")
    if not os.path.isfile(p):
        return None
    nfe, acc, t = [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                nfe.append(float(row["target_nfe"]))
                if row.get("accept_rate") not in ("", None):
                    acc.append(float(row["accept_rate"]))
                t.append(float(row["time_sec"]))
            except (ValueError, KeyError):
                pass
    if not nfe:
        return None
    return {
        "target_nfe": float(np.mean(nfe)),
        "coverage": float(np.mean(acc)) if acc else float("nan"),
        "time": float(np.mean(t)),
    }


def ref_mean_time(ref_dir):
    p = os.path.join(ref_dir, "results.csv")
    if not os.path.isfile(p):
        return None
    ts = []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["time_sec"]))
            except (ValueError, KeyError):
                pass
    return float(np.mean(ts)) if ts else None


def lpips_t_for_config(engine, sweep_root, config, ref_dir):
    """Mean Mask / Boundary LPIPS_t vs target_s50 over shared images."""
    cfg_dir = os.path.join(sweep_root, config)
    m_list, b_list = [], []
    for d in sorted(glob.glob(os.path.join(cfg_dir, "img_*"))):
        idx_name = os.path.basename(d)
        out_p = os.path.join(d, "out.png")
        mask_p = os.path.join(d, "mask.png")
        ref_p = os.path.join(ref_dir, idx_name, "out.png")
        if not all(os.path.isfile(p) for p in (out_p, mask_p, ref_p)):
            continue
        out = _load_png(out_p)
        ref = _load_png(ref_p)
        mask = _load_mask(mask_p)
        mt = engine.compute_vs_target(out, ref, mask)
        m_list.append(mt["masked_lpips_vs_tgt"])
        b_list.append(mt["boundary_lpips_vs_tgt"])
    if not m_list:
        return float("nan"), float("nan"), 0
    return float(np.mean(m_list)), float(np.mean(b_list)), len(m_list)


def verifier_metrics(verif_root, config, far_coverage):
    """Full FreqSpec FAR@far_coverage and AURC for this saliency config."""
    far = float("nan")
    aurc = float("nan")
    vdir = os.path.join(verif_root, config)
    tab = os.path.join(vdir, "table_a_coverage_matched.csv")
    aur = os.path.join(vdir, "aurc_by_selector.csv")
    if os.path.isfile(tab):
        with open(tab) as f:
            for row in csv.DictReader(f):
                if (row["selector"] == "Full FreqSpec"
                        and abs(float(row["coverage"]) - far_coverage) < 1e-6):
                    far = float(row["false_accept_rate"])
    if os.path.isfile(aur):
        with open(aur) as f:
            for row in csv.DictReader(f):
                if row["selector"] == "Full FreqSpec":
                    aurc = float(row["aurc"])
    return far, aurc


def main(args):
    os.makedirs(args.out, exist_ok=True)
    from metrics_extended import RegionMetrics
    engine = RegionMetrics(device=args.device, lpips_net=args.lpips_net,
                           boundary_k=args.boundary_k)

    ref_t = ref_mean_time(args.ref_dir)
    rows = []
    for config, label in CONFIG_ORDER:
        runs = read_runs_csv(args.sweep_root, config)
        if runs is None:
            print(f"[table_b] skip {config} (no runs.csv)")
            continue
        m_lp, b_lp, n = lpips_t_for_config(engine, args.sweep_root, config,
                                           args.ref_dir)
        far, aurc = verifier_metrics(args.verif_root, config, args.far_coverage)
        speedup = (ref_t / runs["time"]) if (ref_t and runs["time"]) else float("nan")
        rows.append({
            "config": config, "label": label, "n": n,
            "target_nfe": runs["target_nfe"], "speedup": speedup,
            "coverage": runs["coverage"], "mask_lpips_t": m_lp,
            "boundary_lpips_t": b_lp, "far": far, "aurc": aurc,
        })
        print(f"[table_b] {config:18s} nfe={runs['target_nfe']:.1f} "
              f"spd={speedup:.2f} cov={runs['coverage']:.3f} "
              f"mLPIPSt={m_lp:.4f} bLPIPSt={b_lp:.4f} FAR={far:.3f} AURC={aurc:.4f}")

    # CSV
    fields = ["config", "label", "n", "target_nfe", "speedup", "coverage",
              "mask_lpips_t", "boundary_lpips_t", "far", "aurc"]
    with open(os.path.join(args.out, "table_b.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})

    _write_latex(rows, os.path.join(args.out, "table_b.tex"), args.far_coverage)
    print(f"[table_b] done -> {args.out}/table_b.csv  +  table_b.tex")


def _write_latex(rows, path, far_cov):
    lines = [
        r"% Requires: \usepackage{booktabs} \usepackage{graphicx}",
        r"\begin{table}[t]\centering",
        r"\caption{\textbf{Ablation of saliency signals for patch-level "
        r"verification (COCO).} All rows share the same draft, images, masks, "
        r"seeds, and Combo~2 calibration; only the saliency signal that "
        r"modulates the acceptance tolerance changes. LPIPS$_t$ is trajectory "
        r"divergence from the 50-step target (mask region / boundary band). "
        r"False-accept and AURC are the Full-FreqSpec verifier's reliability "
        r"under that signal (FAR at "
        + f"{int(far_cov*100)}\\% coverage). "
        r"Wavelet beating Sobel/variance/Laplacian justifies the "
        r"\emph{frequency-guided} design; adding boundary and interior terms "
        r"further lowers boundary LPIPS$_t$ and false accepts.}",
        r"\label{tab:saliency_ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Saliency & Tgt.\ NFE\,$\downarrow$ & Speedup\,$\uparrow$ & "
        r"Cov.\,$\uparrow$ & Mask LPIPS$_t$\,$\downarrow$ & "
        r"Bnd.\ LPIPS$_t$\,$\downarrow$ & FAR\,$\downarrow$ & AURC\,$\downarrow$ \\",
        r"\midrule",
    ]
    # split groups visually: signals | wavelet+components
    group_break_after = {"laplacian"}
    for r in rows:
        bold = (r["config"] == "full")
        def fmt(v, d=4):
            if isinstance(v, float) and not np.isfinite(v):
                return "--"
            return f"{v:.{d}f}"
        cells = [r["label"],
                 fmt(r["target_nfe"], 1), fmt(r["speedup"], 2),
                 fmt(r["coverage"], 3), fmt(r["mask_lpips_t"], 4),
                 fmt(r["boundary_lpips_t"], 4), fmt(r["far"], 3),
                 fmt(r["aurc"], 4)]
        # label already in cells[0]; rebuild line (label + 7 numbers = 8 cols?) ->
        # we have 7 data columns after the label per header (NFE,Speedup,Cov,
        # MaskLPIPSt,BndLPIPSt,FAR,AURC). cells has label+7 = 8 entries. Good.
        if bold:
            body = " & ".join([cells[0]] + [r"\textbf{" + c + "}" for c in cells[1:]])
        else:
            body = " & ".join(cells)
        lines.append(body + r" \\")
        if r["config"] in group_break_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True,
                   help="saliency_ablation_sweep.py --out_root")
    p.add_argument("--ref_dir", required=True,
                   help="target_s50 reference dir (run_target_reference.py)")
    p.add_argument("--verif_root", required=True,
                   help="dir holding <config>/ verifier-analysis subdirs")
    p.add_argument("--out", required=True)
    p.add_argument("--far_coverage", type=float, default=0.5)
    p.add_argument("--boundary_k", type=int, default=8)
    p.add_argument("--lpips_net", type=str, default="alex")
    p.add_argument("--device", type=str, default="cuda")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
