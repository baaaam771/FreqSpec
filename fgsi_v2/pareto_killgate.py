#!/usr/bin/env python
"""
pareto_killgate.py — Reliability-constrained Pareto kill-gate (GPU-free).

Question: "s50 대비 편차가 δ를 넘을 확률을 α 이하로 제한할 때, 가장 빠른
방법은 무엇인가?" per workload (per run in the per-image CSV).

Protocol
- Methods: every method in the CSV + synthetic 's50' (lpips_t = 0,
  per-image time recovered as speedup_i * time_sec_i from any row).
- Calibration/test split: even idx = calibration, odd idx = test.
  Feasibility P(lpips_t > δ) ≤ α is decided on CALIBRATION ONLY; the
  winner's risk/latency are then reported on TEST.
- Risk metrics on test: P(lpips_t > δ), CVaR_α (mean of worst α-fraction
  of lpips_t), mean & p95 latency, mean speedup.
- Paired bootstrap (B=2000) over images: 95% CI for the winner-vs-best-
  FreqSpec speedup difference, and for each method's exceedance prob.
- Also reports the same gate on gt_masked_lpips (δ_gt grid) as a
  robustness check, since lpips_t measures reference reproducibility,
  not GT quality (both views are printed).

Verdict rule (printed at the end): FreqSpec survives the kill-gate iff
some freqspec_* method is the calibration winner for at least one
realistic (δ, α) cell on at least one workload. Otherwise → option B.

Usage:
  python pareto_killgate.py --csv phase2_per_image.csv largemask_per_image.csv \\
      [--deltas 0.005 0.01 0.02 0.05 0.08] [--alphas 0.0 0.01 0.05] [--boot 2000]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    data = defaultdict(dict)  # run -> method -> idx -> dict
    for r in rows:
        d = data[r["run"]].setdefault(r["method"], {})
        d[int(r["idx"])] = {
            "lp": float(r["lpips_t"]),
            "t": float(r["time_sec"]),
            "spd": float(r["speedup"]),
            "gt": float(r["gt_masked_lpips"]) if r.get("gt_masked_lpips")
                  not in (None, "", "nan") else float("nan"),
        }
    # synthesize s50 per run from any existing method
    for run, md in data.items():
        any_m = next(iter(md.values()))
        md["s50_ref"] = {i: {"lp": 0.0,
                             "t": v["spd"] * v["t"],
                             "spd": 1.0,
                             "gt": float("nan")}
                         for i, v in any_m.items()}
    return data


def exceed(vals, delta):
    v = np.asarray(vals, dtype=float)
    return float(np.mean(v > delta)) if len(v) else float("nan")


def cvar(vals, alpha):
    """Mean of the worst max(1, ceil(alpha*n)) values (alpha=0 -> max)."""
    v = np.sort(np.asarray(vals, dtype=float))
    k = max(1, int(np.ceil(max(alpha, 1e-9) * len(v))))
    return float(np.mean(v[-k:]))


def boot_ci(vals, stat, B, rng, alpha_ci=0.05):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    ss = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return (float(np.percentile(ss, 100 * alpha_ci / 2)),
            float(np.percentile(ss, 100 * (1 - alpha_ci / 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--deltas", nargs="+", type=float,
                    default=[0.005, 0.01, 0.02, 0.05, 0.08])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.01, 0.05])
    ap.add_argument("--metric", default="lp", choices=["lp", "gt"],
                    help="risk metric: lp = LPIPS_t vs s50 (default), "
                         "gt = gt_masked_lpips (robustness view; s50_ref "
                         "excluded since its gt risk needs its own rows)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    data = load(args.csv)
    fs_names = lambda md: [m for m in md if m.startswith("freqspec")]
    survivors = []

    for run in sorted(data):
        md = data[run]
        if args.metric == "gt":
            md = {m: d for m, d in md.items() if m != "s50_ref"}
        idxs = sorted(set.intersection(*[set(d) for d in md.values()]))
        calib = [i for i in idxs if i % 2 == 0]
        test = [i for i in idxs if i % 2 == 1]
        print(f"\n================ {run}  (n={len(idxs)}, "
              f"calib={len(calib)}, test={len(test)}, metric={args.metric}) "
              f"================")

        # per-method reference table on TEST
        print(f"{'method':18s} {'spd':>6s} {'t_mean':>7s} {'t_p95':>7s} "
              + " ".join(f"P>{d:g}" .rjust(8) for d in args.deltas)
              + "  CVaR10%")
        for m in sorted(md):
            V = md[m]
            lp = [V[i][args.metric] for i in test]
            spd = float(np.mean([V[i]["spd"] for i in test]))
            tm = float(np.mean([V[i]["t"] for i in test]))
            tp = float(np.percentile([V[i]["t"] for i in test], 95))
            probs = " ".join(f"{exceed(lp, d):8.3f}" for d in args.deltas)
            print(f"{m:18s} {spd:6.3f} {tm:7.3f} {tp:7.3f} {probs} "
                  f"{cvar(lp, 0.10):8.4f}")

        # kill-gate grid: winner chosen on CALIBRATION, reported on TEST
        print(f"\n  kill-gate (winner = fastest calib-feasible; "
              f"test risk in [] with bootstrap 95% CI):")
        for d in args.deltas:
            for a in args.alphas:
                feas = []
                for m, V in md.items():
                    r_cal = exceed([V[i][args.metric] for i in calib], d)
                    if r_cal <= a + 1e-12:
                        t_cal = float(np.mean([V[i]["t"] for i in calib]))
                        feas.append((t_cal, m))
                if not feas:
                    print(f"    δ={d:<6g} α={a:<5g} -> (no feasible method)")
                    continue
                feas.sort()
                t_win, m_win = feas[0]
                V = md[m_win]
                lp_te = [V[i][args.metric] for i in test]
                r_te = exceed(lp_te, d)
                lo, hi = boot_ci(lp_te, lambda v: float(np.mean(v > d)),
                                 args.boot, rng)
                spd_te = float(np.mean([V[i]["spd"] for i in test]))
                # paired speedup diff vs best freqspec on test
                fs = fs_names(md)
                tag = ""
                if fs and not m_win.startswith("freqspec"):
                    best_fs = max(fs, key=lambda m: np.mean(
                        [md[m][i]["spd"] for i in test]))
                    dif = np.array([md[m_win][i]["spd"] - md[best_fs][i]["spd"]
                                    for i in test])
                    dlo, dhi = boot_ci(dif, lambda v: float(np.mean(v)),
                                       args.boot, rng)
                    tag = (f"  Δspd vs {best_fs}: "
                           f"{float(np.mean(dif)):+.3f} [{dlo:+.3f},{dhi:+.3f}]")
                mark = " <<< FreqSpec" if m_win.startswith("freqspec") else ""
                if m_win.startswith("freqspec"):
                    survivors.append((run, d, a))
                print(f"    δ={d:<6g} α={a:<5g} -> {m_win:18s} "
                      f"spd={spd_te:5.3f}  risk_test={r_te:.3f} "
                      f"[{lo:.3f},{hi:.3f}]{tag}{mark}")

    print("\n================ VERDICT ================")
    if survivors:
        print("FreqSpec ON the reliability-constrained frontier at:")
        for run, d, a in survivors:
            print(f"  - {run}: δ={d}, α={a}")
        print("-> Option A remains viable for these (δ, α) regimes "
              "(check they are practically meaningful).")
    else:
        print("FreqSpec dominated at EVERY (δ, α) cell on every workload "
              "-> kill-gate FAILED; proceed with Option B (step-count "
              "router). Run router_signal_check.py next.")


if __name__ == "__main__":
    main()
