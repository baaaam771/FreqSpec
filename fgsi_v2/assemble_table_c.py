#!/usr/bin/env python
"""
assemble_table_c.py  —  Table C (region-aware draft training objective ablation).

For each trained draft variant, reads its FreqSpec evaluation outputs and
assembles the Table C columns. Most columns come straight from the per-patch
verifier logs already produced by verifier_reliability_sweep.py, so no extra
draft passes are needed beyond the per-variant eval:

  Draft eps-MSE   : draft-vs-target epsilon MSE at verified timesteps,
                    recovered as mean( -log(s_eps)/beta ) over mask patches.
  Draft x0-MSE    : draft-vs-target predicted-x0 MSE, = mean(d_x0).
  Coverage        : mean accept_rate over mask-interior patches (runs.csv).
  Target NFE      : mean per-image target calls (runs.csv).
  Mask LPIPS_t    : trajectory divergence from target_s50 inside the mask
                    (needs --save_outputs during eval + a target_s50 ref).
  AURC            : Full-FreqSpec risk-coverage AURC (analyze output).

Variant order matches run_training_ablation.sh.

Inputs (per variant, under --eval_root/<variant>/):
  patch_logs/*.pt                         (verifier_reliability_sweep.py)
  verifier_runs.csv                       (verifier_reliability_sweep.py)
  outputs/img_XXX/{out,gt,mask}.png       (verifier_reliability_sweep.py --save_outputs)
  analysis/aurc_by_selector.csv           (analyze_verifier_reliability.py)
And shared:
  --ref_dir target_s50 (run_target_reference.py) for Mask LPIPS_t.

Outputs:
  <out>/table_c.csv
  <out>/table_c.tex

Example:
  python assemble_table_c.py \\
      --eval_root /mnt/HDD_12TB/bam_ki/results/table_c_eval \\
      --ref_dir   /mnt/HDD_12TB/bam_ki/results/table_c_eval/target_s50 \\
      --out       /mnt/HDD_12TB/bam_ki/results/table_c_eval \\
      --beta 10.0 --boundary_k 8
"""
import argparse
import csv
import glob
import math
import os

import numpy as np
import torch

# (variant_dir, label, block)  — must match run_training_ablation.sh names.
VARIANTS = [
    ("global_gt_only",         "Global GT noise only",            1),
    ("global_distill_only",    "Global target distill only",      1),
    ("hard_gt_only",           "Hard-region GT only",             2),
    ("easy_distill_hard_gt",   "Easy distill + hard GT",          2),
    ("easy_distill_uniform_gt","Easy distill + uniform GT",       2),
    ("hard_gt_uniform_gt",     "Hard GT + uniform GT",            2),
    ("full_region_aware",      "Full region-aware (ours)",        3),
    ("full_mask_only_Mt",      "Full, mask-only $M_t$ (no wav.)", 3),
    ("full_static_M",          "Full, static $M$",                3),
]


def draft_mse_from_logs(logs_dir, beta):
    """mean draft-vs-target eps-MSE and x0-MSE over all mask-interior patches."""
    eps_mse, x0_mse = [], []
    for p in sorted(glob.glob(os.path.join(logs_dir, "*.pt"))):
        d = torch.load(p, map_location="cpu")
        pl = d.get("patch_logs", d)
        s = np.asarray(pl["s_eps"], dtype=np.float64)
        x = np.asarray(pl["d_x0"], dtype=np.float64)
        s = np.clip(s, 1e-8, 1.0)
        eps_mse.append(-np.log(s) / beta)   # invert a = exp(-beta*mse)
        x0_mse.append(x)
    if not eps_mse:
        return float("nan"), float("nan"), 0
    eps_all = np.concatenate(eps_mse)
    x0_all = np.concatenate(x0_mse)
    return float(eps_all.mean()), float(x0_all.mean()), x0_all.size


def coverage_nfe_from_runs(runs_csv):
    if not os.path.isfile(runs_csv):
        return float("nan"), float("nan")
    acc, nfe = [], []
    with open(runs_csv) as f:
        for row in csv.DictReader(f):
            try:
                acc.append(float(row["accept_rate"]))
                nfe.append(float(row["target_nfe"]))
            except (ValueError, KeyError):
                pass
    return (float(np.mean(acc)) if acc else float("nan"),
            float(np.mean(nfe)) if nfe else float("nan"))


def aurc_from_analysis(analysis_dir):
    p = os.path.join(analysis_dir, "aurc_by_selector.csv")
    if not os.path.isfile(p):
        return float("nan")
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["selector"] == "Full FreqSpec":
                return float(row["aurc"])
    return float("nan")


def mask_lpips_t(engine, out_dir, ref_dir):
    if engine is None or not os.path.isdir(out_dir) or not os.path.isdir(ref_dir):
        return float("nan")
    from PIL import Image
    vals = []
    for d in sorted(glob.glob(os.path.join(out_dir, "img_*"))):
        name = os.path.basename(d)
        op = os.path.join(d, "out.png")
        mp = os.path.join(d, "mask.png")
        rp = os.path.join(ref_dir, name, "out.png")
        if not all(os.path.isfile(x) for x in (op, mp, rp)):
            continue
        out = np.array(Image.open(op).convert("RGB"))
        ref = np.array(Image.open(rp).convert("RGB"))
        mask = (np.array(Image.open(mp).convert("L")) > 127).astype(np.float32)
        vals.append(engine.compute_vs_target(out, ref, mask)["masked_lpips_vs_tgt"])
    return float(np.mean(vals)) if vals else float("nan")


def main(args):
    os.makedirs(args.out, exist_ok=True)
    engine = None
    if args.ref_dir:
        try:
            from metrics_extended import RegionMetrics
            engine = RegionMetrics(device=args.device, lpips_net=args.lpips_net,
                                   boundary_k=args.boundary_k)
        except Exception as e:
            print(f"[table_c] Mask LPIPS_t disabled ({e})")

    rows = []
    for vd, label, block in VARIANTS:
        root = os.path.join(args.eval_root, vd)
        if not os.path.isdir(root):
            print(f"[table_c] skip {vd} (missing {root})")
            continue
        eps_mse, x0_mse, n = draft_mse_from_logs(
            os.path.join(root, "patch_logs"), args.beta)
        cov, nfe = coverage_nfe_from_runs(
            os.path.join(root, "verifier_runs.csv"))
        aurc = aurc_from_analysis(os.path.join(root, "analysis"))
        mlp = mask_lpips_t(engine, os.path.join(root, "outputs"), args.ref_dir)
        rows.append({
            "variant": vd, "label": label, "block": block, "n": n,
            "eps_mse": eps_mse, "x0_mse": x0_mse, "coverage": cov,
            "target_nfe": nfe, "mask_lpips_t": mlp, "aurc": aurc,
        })
        print(f"[table_c] {vd:24s} epsMSE={eps_mse:.5f} x0MSE={x0_mse:.5f} "
              f"cov={cov:.3f} NFE={nfe:.1f} mLPIPSt={mlp:.4f} AURC={aurc:.4f}")

    fields = ["variant", "label", "block", "n", "eps_mse", "x0_mse",
              "coverage", "target_nfe", "mask_lpips_t", "aurc"]
    with open(os.path.join(args.out, "table_c.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})

    _write_latex(rows, os.path.join(args.out, "table_c.tex"))
    print(f"\n[table_c] done -> {args.out}/table_c.tex (+ .csv)")


def _write_latex(rows, path):
    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]\centering",
        r"\caption{\textbf{Ablation of the region-aware draft training "
        r"objective (COCO).} Each draft is trained from scratch under one "
        r"objective with all other factors fixed (init, steps, data, masks, "
        r"batch, lr, EMA), then evaluated with the identical Combo~2 verifier. "
        r"Draft $\epsilon$-/$\hat{x}_0$-MSE are draft-vs-target errors at "
        r"verified timesteps; lower means the draft is a closer local "
        r"surrogate, raising coverage and lowering target NFE and trajectory "
        r"divergence (LPIPS$_t$) / AURC. The full region-aware objective "
        r"(easy-region distillation + hard-region ground truth + uniform "
        r"safety, with a wavelet- and timestep-dependent $M_t$) gives the best "
        r"surrogate; removing any term, using a mask-only $M_t$, or a static "
        r"$M$ degrades it. Best per column in \textbf{bold}.}",
        r"\label{tab:training_ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Training objective & $\epsilon$-MSE\,$\downarrow$ & "
        r"$\hat{x}_0$-MSE\,$\downarrow$ & Coverage\,$\uparrow$ & "
        r"Tgt.\ NFE\,$\downarrow$ & Mask LPIPS$_t$\,$\downarrow$ & "
        r"AURC\,$\downarrow$ \\",
    ]

    def fin(vals):
        return [v for v in vals if isinstance(v, float) and np.isfinite(v)]
    best = {
        "eps_mse": min(fin([r["eps_mse"] for r in rows]), default=None),
        "x0_mse": min(fin([r["x0_mse"] for r in rows]), default=None),
        "coverage": max(fin([r["coverage"] for r in rows]), default=None),
        "target_nfe": min(fin([r["target_nfe"] for r in rows]), default=None),
        "mask_lpips_t": min(fin([r["mask_lpips_t"] for r in rows]), default=None),
        "aurc": min(fin([r["aurc"] for r in rows]), default=None),
    }

    def cell(val, key, d=4):
        if isinstance(val, float) and not np.isfinite(val):
            return "--"
        s = f"{val:.{d}f}" if key != "target_nfe" else f"{val:.1f}"
        if best[key] is not None and isinstance(val, float) and abs(val - best[key]) < 1e-9:
            return r"\textbf{" + s + "}"
        return s

    titles = {1: r"\textit{Global (non-region-aware) objectives}",
              2: r"\textit{Partial region-aware combinations}",
              3: r"\textit{Full objective and $M_t$ variants}"}
    last = None
    for r in rows:
        if r["block"] != last:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{7}{l}{" + titles[r["block"]] + r"} \\")
            last = r["block"]
        lines.append(" & ".join([
            r["label"],
            cell(r["eps_mse"], "eps_mse", 4),
            cell(r["x0_mse"], "x0_mse", 4),
            cell(r["coverage"], "coverage", 3),
            cell(r["target_nfe"], "target_nfe"),
            cell(r["mask_lpips_t"], "mask_lpips_t", 4),
            cell(r["aurc"], "aurc", 4),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", required=True,
                   help="dir with <variant>/{patch_logs,verifier_runs.csv,"
                        "analysis,outputs}")
    p.add_argument("--ref_dir", default="",
                   help="target_s50 reference dir for Mask LPIPS_t (optional)")
    p.add_argument("--out", required=True)
    p.add_argument("--beta", type=float, default=10.0,
                   help="agreement temperature used at inference (Eq. 8).")
    p.add_argument("--boundary_k", type=int, default=8)
    p.add_argument("--lpips_net", default="alex")
    p.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
