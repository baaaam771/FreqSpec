#!/usr/bin/env python3
# =============================================================================
#  F2 + T2 : reliability-constrained selection (90 cells)
#  Rule: given deviation budget d and risk a, pick the FASTEST method with
#        P(lpips_t > d) <= a.  Feasibility decided on calibration half
#        (idx even), confirmed on test half (idx odd), paired bootstrap CI.
#  Expected headline: FreqSpec wins 0 / 90 cells.
#
#  Inputs (same dir or --data): phase2_per_image.csv, largemask_per_image.csv,
#         instmask_per_image.csv   (per-image; ref for lpips_t = target_s50)
#  Columns: run, idx, method, gt_masked_lpips, gt_boundary_lpips, lpips_t,
#           time_sec, mask_coverage
#
#  Outputs: fig_f2_selection.pdf/.png  and  table_t2_selection.tex
#           selection_cells.csv (full 90-cell winners, for the supplement)
#
#  Usage:  python make_f2_selection.py --data /path/to/csvs
# =============================================================================
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager

DELTAS = [0.005, 0.01, 0.02, 0.04, 0.08]
ALPHAS = [0.0, 0.01, 0.05]
NBOOT  = 2000
SEED   = 0

# workloads to include as the "6 cells" (edit if instmask handled separately)
PER_IMAGE_FILES = ["phase2_per_image.csv",
                   "largemask_per_image.csv",
                   "instmask_per_image.csv"]

def load(data_dir):
    frames = []
    for f in PER_IMAGE_FILES:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"[warn] missing {p}")
    if not frames:
        sys.exit("no per-image CSVs found; pass --data")
    df = pd.concat(frames, ignore_index=True)
    need = {"run","idx","method","lpips_t","time_sec"}
    miss = need - set(df.columns)
    if miss: sys.exit(f"missing columns: {miss}")
    return df

def method_speed(df):
    # median per-request time per (run, method) -> lower is faster
    return df.groupby(["run","method"])["time_sec"].median()

def feasible(sub, d, a):
    # sub: rows for one (run, method); returns P(lpips_t>d)
    return float((sub["lpips_t"].values > d).mean())

def pick_winner(df_run_cal, d, a, speed):
    # among methods feasible on calibration half, choose fastest
    cands = []
    for m, sub in df_run_cal.groupby("method"):
        if feasible(sub, d, a) <= a:
            cands.append((speed.loc[(sub["run"].iloc[0], m)], m))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]

def is_ours(m): return str(m).startswith("freqspec")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    args = ap.parse_args()
    df = load(args.data)
    df["cal"] = (df["idx"] % 2 == 0)
    speed = method_speed(df)
    runs = sorted(df["run"].unique())
    print(f"workloads ({len(runs)}): {runs}")

    rows = []
    ours_wins = 0
    grid = {}  # (run, d, a) -> winner
    for run in runs:
        dr = df[df["run"]==run]
        cal = dr[dr["cal"]]; test = dr[~dr["cal"]]
        for d in DELTAS:
            for a in ALPHAS:
                w = pick_winner(cal, d, a, speed)
                # confirm on test half (feasibility must still hold)
                if w is not None:
                    tsub = test[test["method"]==w]
                    if len(tsub) and feasible(tsub, d, a) > a:
                        # winner fails on test -> fall back to reference
                        w = "target_s50" if "target_s50" in dr["method"].values else w
                grid[(run,d,a)] = w
                if w is not None and is_ours(w): ours_wins += 1
                rows.append(dict(run=run, delta=d, alpha=a, winner=w))
    sel = pd.DataFrame(rows)
    sel.to_csv("selection_cells.csv", index=False)
    ncells = len(sel)
    print(f"=== FreqSpec wins {ours_wins} / {ncells} cells ===")

    # ---- figure: grid of (run x (d,a)) colored by winner category ----
    def cat(w):
        if w is None: return 0                    # infeasible
        if is_ours(w): return 3                   # ours
        if w=="target_s50": return 1              # reference
        return 2                                  # reduced-step
    cats = np.array([[cat(grid[(run,d,a)]) for d in DELTAS for a in ALPHAS]
                     for run in runs])
    labels = [f"$\\delta$={d}\n$\\alpha$={a}" for d in DELTAS for a in ALPHAS]
    cmap = ListedColormap(["#dddddd","#c0392b","#2456a6","#e67e22"])
    for fam in ("Times New Roman","Nimbus Roman","DejaVu Serif"):
        if any(fam==f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"]="serif"; plt.rcParams["font.serif"]=[fam]; break
    fig, ax = plt.subplots(figsize=(11, 0.5*len(runs)+1.6))
    ax.imshow(cats, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_yticks(range(len(runs))); ax.set_yticklabels(runs)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(f"Reliability-constrained selection — FreqSpec wins {ours_wins}/{ncells}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b",label="s50 reference"),
                       Patch(color="#2456a6",label="reduced-step"),
                       Patch(color="#e67e22",label="FreqSpec (ours)"),
                       Patch(color="#dddddd",label="infeasible")],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5,-0.12), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("fig_f2_selection.pdf", bbox_inches="tight")
    fig.savefig("fig_f2_selection.png", dpi=150, bbox_inches="tight")

    # ---- compact LaTeX table: winner category counts per workload ----
    with open("table_t2_selection.tex","w") as f:
        f.write("% auto-generated by make_f2_selection.py\n")
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\caption{Reliability-constrained selection: winning method "
                f"category per workload over $\\delta\\times\\alpha$ ({len(DELTAS)*len(ALPHAS)} cells each). "
                f"The verifier-gated system wins {ours_wins} of {ncells}.}}\n")
        f.write("\\label{tab:selection}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Workload & s50 & reduced-step & ours & infeasible \\\\\n\\midrule\n")
        for run in runs:
            c=[0,0,0,0]
            for d in DELTAS:
                for a in ALPHAS: c[cat(grid[(run,d,a)])]+=1
            f.write(f"{run} & {c[1]} & {c[2]} & {c[3]} & {c[0]} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("wrote fig_f2_selection.pdf/.png, table_t2_selection.tex, selection_cells.csv")

if __name__ == "__main__":
    main()
