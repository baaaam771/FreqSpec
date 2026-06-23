#!/usr/bin/env python
"""
sr_frequency_ablation.py — how much should the SR verifier rely on frequency?

The reliability analysis showed that for SR a pure wavelet saliency selector has
*lower* AURC than the deployed Full FreqSpec rule (the opposite of inpainting,
where frequency is only a strictness prior). This script quantifies that finding
as a controllable contribution, entirely offline from the existing 17.6M-patch
logs (no diffusion re-run):

    conf_lambda = (1 - lambda) * rank(base) + lambda * rank(1 - A_wav)

where base is the deployable signal (Full FreqSpec weight w, or epsilon
agreement), and A_wav is the pure wavelet saliency. We rank-normalize both so the
mix is scale-fair, sweep lambda in [0,1], and report AURC(lambda). lambda=0 is the
base rule as deployed, lambda=1 is wavelet-only. If AURC dips below the base at
some lambda*>0, mixing in frequency improves the deployable SR verifier, and
lambda* is the recommended frequency weight.

Outputs (under --out_dir):
    freq_ablation.csv      AURC vs lambda for each base
    freq_ablation.png      AURC(lambda) curves
    summary printed to console (base AURC, best lambda, wavelet AURC)

Usage:
    python sr_frequency_ablation.py \
        --logs_dir /mnt/HDD_12TB/bam_ki/results/sr_div2k_100k/reliability/patch_logs \
        --out_dir  /mnt/HDD_12TB/bam_ki/results/sr_div2k_100k/freq_ablation
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyze_verifier_reliability import (load_logs, merge, risk_coverage_curve,
                                          compute_aurc)


def rank_norm(x):
    """Percentile rank in [0,1] (ties broken by position); scale-free."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks / max(x.size - 1, 1)


def aurc_for_conf(conf, risk, coverages):
    cov, risks = risk_coverage_curve(conf, risk, coverages)
    return compute_aurc(cov, risks)


def main(args):
    items = load_logs(args.logs_dir)
    merged = merge(items)
    if "wav" not in merged:
        sys.exit("[freq-abl] logs have no pure-wavelet 'wav' field; cannot run "
                 "frequency ablation (re-run sweep so 'wav' is logged).")

    risk = merged["d_x0"]
    n = risk.size
    if args.subsample and n > args.subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=args.subsample, replace=False)
        merged = {k: v[idx] for k, v in merged.items()}
        risk = merged["d_x0"]
        print(f"[freq-abl] subsampled {n} -> {risk.size} patches")
    else:
        print(f"[freq-abl] using all {n} patches")

    cf = rank_norm(1.0 - merged["wav"])             # frequency confidence (low HF = safe)
    bases = {
        "Full (w)": rank_norm(merged["w"]),
        "Epsilon": rank_norm(merged["s_eps"]),
    }
    coverages = np.linspace(0.05, 1.0, args.n_cov)
    lambdas = np.linspace(0.0, 1.0, args.n_lambda)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    results = {}
    for bname, cb in bases.items():
        aurcs = []
        for lam in lambdas:
            conf = (1.0 - lam) * cb + lam * cf
            aurcs.append(aurc_for_conf(conf, risk, coverages))
        aurcs = np.asarray(aurcs)
        results[bname] = aurcs
        i_best = int(np.argmin(aurcs))
        base_aurc, wav_aurc, best_aurc = aurcs[0], aurcs[-1], aurcs[i_best]
        impr = 100.0 * (base_aurc - best_aurc) / base_aurc
        print(f"\n[{bname}] AURC(lambda):")
        print(f"  lambda=0  (base)     : {base_aurc:.5f}")
        print(f"  lambda=1  (wavelet)  : {wav_aurc:.5f}")
        print(f"  lambda*={lambdas[i_best]:.2f} (best)  : {best_aurc:.5f}  "
              f"({impr:+.1f}% vs base)")
        for lam, a in zip(lambdas, aurcs):
            rows.append({"base": bname, "lambda": f"{lam:.3f}", "aurc": f"{a:.6f}",
                         "is_best": int(np.isclose(a, best_aurc))})

    csv_out = os.path.join(args.out_dir, "freq_ablation.csv")
    with open(csv_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["base", "lambda", "aurc", "is_best"])
        wr.writeheader(); wr.writerows(rows)
    print(f"\n[freq-abl] wrote {csv_out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4.2))
        for bname, aurcs in results.items():
            plt.plot(lambdas, aurcs, marker="o", ms=4, label=f"{bname} + wavelet")
            ib = int(np.argmin(aurcs))
            plt.scatter([lambdas[ib]], [aurcs[ib]], s=90, facecolors="none",
                        edgecolors="red", zorder=5)
        plt.xlabel(r"frequency mixing weight $\lambda$  (0=base, 1=wavelet-only)")
        plt.ylabel("AURC  (lower = better verifier)")
        plt.title("SR verifier: mixing wavelet saliency into the acceptance signal")
        plt.legend(fontsize=9); plt.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(args.out_dir, "freq_ablation.png")
        plt.savefig(png, dpi=160); print(f"[freq-abl] wrote {png}")
    except Exception as e:
        print(f"[freq-abl] plot skipped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_lambda", type=int, default=11)
    ap.add_argument("--n_cov", type=int, default=40)
    ap.add_argument("--subsample", type=int, default=4000000,
                    help="cap patches for speed (0 = use all)")
    main(ap.parse_args())
