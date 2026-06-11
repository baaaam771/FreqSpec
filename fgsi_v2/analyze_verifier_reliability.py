#!/usr/bin/env python
"""
analyze_verifier_reliability.py  —  Table A (verifier reliability + risk-coverage).

Consumes the per-patch verifier logs produced by verifier_reliability_sweep.py
(one .pt file per image-seed, each holding a dict of 1-D tensors:
    d_x0, s_eps, w, saliency, t_norm
plus scalar meta image_id / seed) and produces, exactly as specified in the
experiment design (Q3):

  1. Risk-coverage curves comparing several patch-selection rules:
        Random (averaged over repeats), Wavelet-only, Epsilon-agreement,
        x0-gate, Full FreqSpec (blend weight), Oracle.
  2. AURC (area under risk-coverage) for each rule.
  3. Coverage-matched risk table at 30 / 50 / 70 percent coverage,
     with accepted-error, rejected-error, error-ratio, false-accept-rate,
     and bad-patch-recall.
  4. Timestep-region breakdown (early-verified / drift-sensitive / late).
  5. Micro (all patches pooled) and macro (per image-seed, with 95 percent
     bootstrap CI) summaries.

Outputs (under --out_dir):
    table_a_coverage_matched.csv     main paper table
    table_a.tex                      LaTeX version of the coverage-matched table
    aurc_by_selector.csv             AURC per selection rule
    risk_coverage.csv                full curve points (per selector)
    timestep_region.csv              per-region accepted/rejected/FAR/AURC
    risk_coverage.png                curve figure (if matplotlib available)
    summary.txt                      human-readable summary

The "risk" of an accepted patch is its predicted-x0 disagreement d_x0(p) against
the target (paper Eq. 11). A good verifier accepts low-risk patches first, so its
risk-coverage curve stays low and its AURC is small.

Usage:
    python analyze_verifier_reliability.py \\
        --logs_dir /mnt/HDD_12TB/bam_ki/results/verifier_coco/patch_logs \\
        --out_dir  /mnt/HDD_12TB/bam_ki/results/verifier_coco/analysis \\
        --bad_quantile 0.8 --random_repeats 20 --bootstrap 2000
"""
import argparse
import csv
import glob
import math
import os

import numpy as np
import torch


# ====================================================================
# Loading
# ====================================================================
def load_logs(logs_dir):
    """Load all per-image-seed .pt logs. Returns a list of dicts, each with
    numpy arrays d_x0/s_eps/w/saliency/t_norm and meta image_id/seed."""
    paths = sorted(glob.glob(os.path.join(logs_dir, "*.pt")))
    if not paths:
        raise SystemExit(f"no .pt logs found under {logs_dir}")
    items = []
    for p in paths:
        d = torch.load(p, map_location="cpu")
        logs = d["patch_logs"] if "patch_logs" in d else d
        rec = {k: np.asarray(logs[k], dtype=np.float64)
               for k in ("d_x0", "s_eps", "w", "saliency", "t_norm")}
        if rec["d_x0"].size == 0:
            continue
        # optional pure-wavelet component (only if logged separately)
        if "wav" in logs and np.asarray(logs["wav"]).size == rec["d_x0"].size:
            rec["wav"] = np.asarray(logs["wav"], dtype=np.float64)
        rec["image_id"] = d.get("image_id", os.path.basename(p))
        rec["seed"] = d.get("seed", -1)
        items.append(rec)
    if not items:
        raise SystemExit("all logs were empty")
    print(f"[analyze] loaded {len(items)} image-seed logs, "
          f"{sum(r['d_x0'].size for r in items)} mask-interior patches total")
    return items


def merge(items, keys=("d_x0", "s_eps", "w", "saliency", "t_norm")):
    out = {k: np.concatenate([r[k] for r in items], axis=0) for k in keys}
    if all("wav" in r for r in items):
        out["wav"] = np.concatenate([r["wav"] for r in items], axis=0)
    return out


# ====================================================================
# Confidence selectors  (higher = more confident the draft is safe here)
# ====================================================================
def build_selectors(merged, rng):
    """Return {name: confidence_array}. Risk is d_x0, so a perfect selector
    ranks by -d_x0 (oracle). The honest deployable baselines are the saliency
    prior and epsilon agreement; full is the proposed rule.

    NOTE on labels:
      "Saliency (1-A)" is 1 - the *combined* saliency A that was logged at run
      time (A = wavelet + boundary + interior under Combo 2). It is NOT a pure
      wavelet signal, so it must not be used to justify the "frequency-guided"
      claim -- that comparison belongs in Table B. A genuine wavelet-only
      column appears automatically only if the logs carry a separate "wav"
      (pure A_wav) field; see verifier_reliability_sweep.py --log_components.

      The x0-gate uses the target's own x0, so ranking by -d_x0 is identical to
      the oracle on this risk metric. We therefore report a single
      "Oracle (x0 upper bound)" row instead of two identical lines.
    """
    err = merged["d_x0"]
    sel = {
        "Random":                  rng.random(err.shape),
        "Saliency (1-A)":          1.0 - merged["saliency"],
        "Epsilon agreement":       merged["s_eps"],
        "Full FreqSpec":           merged["w"],
        "Oracle (x0 upper bound)": -err,
    }
    # genuine pure-wavelet column, only when separately logged
    if "wav" in merged and np.asarray(merged["wav"]).size:
        ordered = {
            "Random": sel["Random"],
            "Wavelet only (A_wav)": 1.0 - np.asarray(merged["wav"]),
            "Saliency (1-A)": sel["Saliency (1-A)"],
            "Epsilon agreement": sel["Epsilon agreement"],
            "Full FreqSpec": sel["Full FreqSpec"],
            "Oracle (x0 upper bound)": sel["Oracle (x0 upper bound)"],
        }
        return ordered
    return sel


# ====================================================================
# Risk-coverage  (Q3 sec 9 / 10)
# ====================================================================
def risk_coverage_curve(confidence, error, coverages):
    c = np.asarray(confidence, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(c) & np.isfinite(e)
    c, e = c[finite], e[finite]
    if c.size == 0:
        raise ValueError("no valid patches")
    order = np.argsort(-c)               # most confident first
    e_sorted = e[order]
    cum = np.cumsum(e_sorted)
    counts = np.arange(1, e_sorted.size + 1)
    cum_risk = cum / counts
    risks = []
    n = e_sorted.size
    for cov in coverages:
        idx = int(math.ceil(float(cov) * n)) - 1
        idx = max(0, min(idx, n - 1))
        risks.append(float(cum_risk[idx]))
    return np.asarray(coverages, dtype=np.float64), np.asarray(risks)


def compute_aurc(coverages, risks):
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(risks, coverages))


# ====================================================================
# Coverage-matched statistics at a fixed coverage  (Q3 sec 14)
# ====================================================================
def coverage_matched_stats(confidence, error, coverage, bad_quantile):
    c = np.asarray(confidence, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(c) & np.isfinite(e)
    c, e = c[finite], e[finite]
    n = e.size
    k = max(1, int(math.ceil(coverage * n)))
    order = np.argsort(-c)
    accepted_idx = order[:k]
    rejected_idx = order[k:]
    acc_err = float(e[accepted_idx].mean())
    rej_err = float(e[rejected_idx].mean()) if rejected_idx.size else float("nan")
    bad_thr = float(np.quantile(e, bad_quantile))
    bad_mask = e >= bad_thr
    # false-accept rate: of accepted patches, fraction that are "bad"
    far = float((e[accepted_idx] >= bad_thr).mean())
    # bad-patch recall: of all bad patches, fraction wrongly accepted
    n_bad = int(bad_mask.sum())
    accepted_bool = np.zeros(n, dtype=bool)
    accepted_bool[accepted_idx] = True
    bad_recall = float((accepted_bool & bad_mask).sum() / max(n_bad, 1))
    return {
        "coverage": coverage,
        "accepted_error": acc_err,
        "rejected_error": rej_err,
        "error_ratio": (rej_err / acc_err) if acc_err > 0 else float("nan"),
        "false_accept_rate": far,
        "bad_patch_recall": bad_recall,
        "overall_error": float(e.mean()),
    }


# ====================================================================
# Timestep regions  (Q3 sec 15)
# ====================================================================
def region_masks(t_norm):
    return {
        "early_verified":  (t_norm > 0.55) & (t_norm <= 0.70),
        "drift_sensitive": (t_norm > 0.35) & (t_norm <= 0.55),
        "late":            (t_norm <= 0.35),
    }


# ====================================================================
# Macro (per image-seed) bootstrap CI  (Q3 sec 16)
# ====================================================================
def macro_bootstrap(items, selector_name, coverage, bad_quantile,
                    n_boot, rng):
    """Compute the coverage-matched accepted-error per image-seed for the
    given selector, then bootstrap a 95 percent CI over image-seeds."""
    per_image = []
    for r in items:
        conf = _confidence_for(r, selector_name, rng)
        st = coverage_matched_stats(conf, r["d_x0"], coverage, bad_quantile)
        if np.isfinite(st["accepted_error"]):
            per_image.append(st["accepted_error"])
    per_image = np.asarray(per_image, dtype=np.float64)
    if per_image.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    boots = []
    m = per_image.size
    for _ in range(n_boot):
        sample = per_image[rng.integers(0, m, m)]
        boots.append(sample.mean())
    boots = np.asarray(boots)
    return (float(per_image.mean()),
            (float(np.percentile(boots, 2.5)),
             float(np.percentile(boots, 97.5))))


def _confidence_for(rec, name, rng):
    err = rec["d_x0"]
    if name == "Random":
        return rng.random(err.shape)
    if name == "Wavelet only (A_wav)":
        return 1.0 - np.asarray(rec["wav"])
    if name == "Saliency (1-A)":
        return 1.0 - rec["saliency"]
    if name == "Epsilon agreement":
        return rec["s_eps"]
    if name == "Oracle (x0 upper bound)":
        return -err
    if name == "Full FreqSpec":
        return rec["w"]
    raise ValueError(name)


# ====================================================================
# Main
# ====================================================================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    items = load_logs(args.logs_dir)
    merged = merge(items)
    n_patches = merged["d_x0"].size

    coverage_grid = np.linspace(0.05, 1.0, 20)
    selectors = build_selectors(merged, rng)

    # ---- 1+2: risk-coverage curves + AURC ----
    curves, aurc = {}, {}
    for name, conf in selectors.items():
        if name == "Random":
            # average over repeats for a stable Random curve (Q3 sec 13)
            stacks = []
            for _ in range(args.random_repeats):
                _, risk = risk_coverage_curve(
                    rng.random(merged["d_x0"].shape), merged["d_x0"],
                    coverage_grid)
                stacks.append(risk)
            risk = np.mean(np.stack(stacks), axis=0)
            cov = coverage_grid
        else:
            cov, risk = risk_coverage_curve(conf, merged["d_x0"], coverage_grid)
        curves[name] = (cov, risk)
        aurc[name] = compute_aurc(cov, risk)

    with open(os.path.join(args.out_dir, "aurc_by_selector.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["selector", "aurc"])
        for name in selectors:
            w.writerow([name, f"{aurc[name]:.6f}"])

    with open(os.path.join(args.out_dir, "risk_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["selector", "coverage", "risk"])
        for name, (cov, risk) in curves.items():
            for cc, rr in zip(cov, risk):
                w.writerow([name, f"{cc:.4f}", f"{rr:.6f}"])

    # ---- 3: coverage-matched table at 30/50/70 ----
    cov_points = [0.30, 0.50, 0.70]
    rows = []
    for name in selectors:
        conf = selectors[name]
        for cov in cov_points:
            st = coverage_matched_stats(conf, merged["d_x0"], cov,
                                        args.bad_quantile)
            macro_mean, (lo, hi) = macro_bootstrap(
                items, name, cov, args.bad_quantile, args.bootstrap, rng)
            rows.append({
                "selector": name, "coverage": cov,
                "accepted_error_micro": st["accepted_error"],
                "rejected_error_micro": st["rejected_error"],
                "error_ratio": st["error_ratio"],
                "false_accept_rate": st["false_accept_rate"],
                "bad_patch_recall": st["bad_patch_recall"],
                "accepted_error_macro": macro_mean,
                "macro_ci_lo": lo, "macro_ci_hi": hi,
                "aurc": aurc[name],
            })

    fields = ["selector", "coverage", "accepted_error_micro",
              "rejected_error_micro", "error_ratio", "false_accept_rate",
              "bad_patch_recall", "accepted_error_macro",
              "macro_ci_lo", "macro_ci_hi", "aurc"]
    with open(os.path.join(args.out_dir, "table_a_coverage_matched.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})

    _write_latex(rows, cov_points, selectors, aurc,
                 os.path.join(args.out_dir, "table_a.tex"))

    # ---- 4: timestep-region breakdown (Full FreqSpec, fixed 50% coverage) ----
    regions = region_masks(merged["t_norm"])
    with open(os.path.join(args.out_dir, "timestep_region.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "selector", "n_patches", "coverage",
                    "accepted_error", "rejected_error",
                    "false_accept_rate", "aurc"])
        for rname, rmask in regions.items():
            if rmask.sum() == 0:
                continue
            err_r = merged["d_x0"][rmask]
            region_selectors = [s for s in (
                "Random", "Wavelet only (A_wav)", "Saliency (1-A)",
                "Epsilon agreement", "Full FreqSpec",
                "Oracle (x0 upper bound)") if s in selectors]
            for sname in region_selectors:
                conf_r = selectors[sname][rmask]
                st = coverage_matched_stats(conf_r, err_r, 0.50,
                                            args.bad_quantile)
                cov_r, risk_r = risk_coverage_curve(conf_r, err_r,
                                                    coverage_grid)
                w.writerow([rname, sname, int(rmask.sum()), 0.50,
                            f"{st['accepted_error']:.6f}",
                            f"{st['rejected_error']:.6f}",
                            f"{st['false_accept_rate']:.4f}",
                            f"{compute_aurc(cov_r, risk_r):.6f}"])

    # ---- plot ----
    _maybe_plot(curves, aurc, os.path.join(args.out_dir, "risk_coverage.png"))

    # ---- summary ----
    _write_summary(args, items, n_patches, aurc, rows, cov_points,
                   os.path.join(args.out_dir, "summary.txt"))

    print(f"[analyze] done -> {args.out_dir}")
    print(f"[analyze] AURC: " + "  ".join(
        f"{k}={v:.4f}" for k, v in aurc.items()))


def _write_latex(rows, cov_points, selectors, aurc, path):
    by = {(r["selector"], r["coverage"]): r for r in rows}
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Reliability of patch-level draft verification (COCO). "
        r"Risk is the coverage-matched predicted-$\hat{x}_0$ disagreement of "
        r"accepted patches; lower accepted error, higher error ratio, and "
        r"lower false-accept rate (FAR, fraction of accepted patches in the "
        r"top-20\% error tail) indicate a better verifier. AURC integrates "
        r"risk over coverage $[0.05,1.0]$. ``Saliency $(1{-}A)$'' uses the "
        r"combined frequency-and-boundary saliency as a standalone selector; "
        r"its near-random AURC shows that saliency is a strictness prior, not "
        r"a direct disagreement predictor. ``Oracle ($\hat{x}_0$ upper "
        r"bound)'' ranks by true $\hat{x}_0$ disagreement and equals an "
        r"$\hat{x}_0$-gate oracle on this metric. Full FreqSpec combines "
        r"agreement and $\hat{x}_0$ tests and approaches the oracle.}",
        r"\label{tab:verifier_reliability}",
        r"\small",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Method & Cov. & Acc.\ err.$\downarrow$ & Rej.\ err.$\uparrow$ & "
        r"Ratio$\uparrow$ & FAR$\downarrow$ & AURC$\downarrow$ \\",
        r"\midrule",
    ]
    for name in selectors:
        tex_name = name.replace("_", r"\_")
        for cov in cov_points:
            r = by[(name, cov)]
            lines.append(
                f"{tex_name} & {int(cov*100)}\\% & "
                f"{r['accepted_error_micro']:.4f} & "
                f"{r['rejected_error_micro']:.4f} & "
                f"{r['error_ratio']:.2f} & "
                f"{r['false_accept_rate']:.3f} & "
                f"{aurc[name]:.4f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _maybe_plot(curves, aurc, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[analyze] matplotlib unavailable, skipping plot")
        return
    plt.figure(figsize=(6, 4.2))
    style = {"Random": ":", "Oracle": "--"}
    for name, (cov, risk) in curves.items():
        plt.plot(cov, risk, style.get(name, "-"),
                 label=f"{name} (AURC={aurc[name]:.4f})", linewidth=1.8)
    plt.xlabel("Coverage (fraction of mask patches drafted)")
    plt.ylabel(r"Risk = accepted-patch $\hat{x}_0$ disagreement")
    plt.title("Risk-coverage of patch-level draft verification")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[analyze] wrote {path}")


def _write_summary(args, items, n_patches, aurc, rows, cov_points, path):
    lines = []
    lines.append("Verifier reliability analysis (Table A)")
    lines.append("=" * 50)
    lines.append(f"image-seed logs : {len(items)}")
    lines.append(f"mask patches    : {n_patches}")
    lines.append(f"bad quantile    : {args.bad_quantile}")
    lines.append("")
    lines.append("AURC (lower is better):")
    for k, v in sorted(aurc.items(), key=lambda kv: kv[1]):
        lines.append(f"  {k:20s} {v:.5f}")
    lines.append("")
    lines.append("Coverage-matched accepted error (micro):")
    for cov in cov_points:
        lines.append(f"  coverage {int(cov*100)}%:")
        for r in rows:
            if r["coverage"] == cov:
                lines.append(
                    f"    {r['selector']:20s} acc={r['accepted_error_micro']:.5f}"
                    f"  rej={r['rejected_error_micro']:.5f}"
                    f"  ratio={r['error_ratio']:.2f}"
                    f"  FAR={r['false_accept_rate']:.3f}")
    lines.append("")
    lines.append("Interpretation: a good verifier has Full FreqSpec accepted")
    lines.append("error well below Random and close to Oracle, with a high")
    lines.append("rejected/accepted error ratio and a low false-accept rate.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", required=True,
                   help="Directory of per-image-seed .pt patch logs.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bad_quantile", type=float, default=0.8,
                   help="Top fraction of error treated as 'bad' for FAR.")
    p.add_argument("--random_repeats", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())