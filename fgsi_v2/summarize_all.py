#!/usr/bin/env python
"""All-results summary + Pareto curve update for FreqSpec-Inpaint paper.

Outputs:
  - /mnt/HDD_12TB/bam_ki/results/all_results.csv
  - /mnt/HDD_12TB/bam_ki/results/pareto_curve_v2.png
  - /mnt/HDD_12TB/bam_ki/results/ablation_table.csv
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# Master results dictionary (모든 측정 결과)
# Columns: (model, dataset, setup_label, speedup, PSNR, SSIM, LPIPS,
#          accept_rate, NFE_target, category)
#   category: "main" (Pareto), "ablation_K", "ablation_tol",
#             "ablation_saliency", "ablation_boundary",
#             "ablation_tstart", "ablation_patch", "cross_domain"
# ===================================================================
ALL_RESULTS = [
    # (model, dataset, setup, speedup, PSNR, SSIM, LPIPS, accept, NFE, category, label_short)
    # Main Pareto curve points
    ("SD2",  "Places2", "default (K=3, tol=0.03/0.3)", 1.28, 22.83, 0.956, 0.064, 0.40, 39.5, "main",  "SD2 P2 default"),
    ("SD2",  "Places2", "mid (K=3, tol=0.02/0.15)",    1.04, 31.08, 0.990, 0.015, 0.15, 48.0, "main",  "SD2 P2 mid"),
    ("SD2",  "Places2", "strict (K=3, tol=0.01/0.1)",  1.02, 36.74, 0.996, 0.007, 0.08, 49.2, "main",  "SD2 P2 strict"),

    ("SDXL", "Places2", "default (K=3, tol=0.03/0.3)", 1.35, 22.08, 0.948, 0.074, 0.37, 37.4, "main",  "SDXL P2 default"),
    ("SDXL", "Places2", "mid (K=3, tol=0.02/0.15)",    1.11, 26.22, 0.974, 0.038, 0.18, 45.6, "main",  "SDXL P2 mid"),
    ("SDXL", "Places2", "strict (K=3, tol=0.01/0.1)",  1.05, 25.88, 0.977, 0.029, 0.11, 47.6, "main",  "SDXL P2 strict"),

    ("SDXL", "FFHQ",    "default (K=3, tol=0.03/0.3)", 1.62, 22.44, 0.934, 0.085, 0.60, 31.0, "main",  "SDXL FFHQ default"),
    ("SDXL", "FFHQ",    "mid (K=3, tol=0.02/0.15)",    1.36, 25.23, 0.960, 0.043, 0.40, 37.2, "main",  "SDXL FFHQ mid"),
    ("SDXL", "FFHQ",    "strict (K=3, tol=0.01/0.1)",  1.12, 27.58, 0.972, 0.025, 0.29, 44.8, "main",  "SDXL FFHQ strict"),

    # Ablation: K sweep (FFHQ)
    ("SDXL", "FFHQ",    "K=1 (tol=0.03/0.3)",          1.00, 24.23, 0.955, 0.053, 0.73, 50.0, "ablation_K", "K=1"),
    # K=3 default already in main
    ("SDXL", "FFHQ",    "K=5 (tol=0.03/0.3)",          1.99, 22.16, 0.934, 0.075, 0.61, 25.2, "ablation_K", "K=5"),

    # Ablation: K sweep (Places2)
    ("SDXL", "Places2", "K=1 (tol=0.03/0.3)",          1.00, 23.35, 0.957, 0.057, 0.47, 50.0, "ablation_K", "K=1"),
    ("SDXL", "Places2", "K=5 (tol=0.03/0.3)",          1.44, 20.25, 0.935, 0.089, 0.28, 35.6, "ablation_K", "K=5"),

    # Ablation: Saliency on/off (Places2 SDXL)
    ("SDXL", "Places2", "uniform saliency (no wavelet)", 1.31, 20.62, 0.939, 0.085, 0.37, 38.4, "ablation_saliency", "no saliency"),

    # Ablation: Boundary on/off
    ("SDXL", "Places2", "no boundary (boundary_weight=0)", 1.20, 21.73, 0.945, 0.073, 0.28, 42.4, "ablation_boundary", "no boundary"),

    # Ablation: t_spec_start (Places2 SDXL)
    ("SDXL", "Places2", "t_spec_start=0.5",            1.07, 26.23, 0.973, 0.037, 0.22, 46.8, "ablation_tstart", "t_start=0.5"),
    ("SDXL", "Places2", "t_spec_start=0.9",            1.64, 17.12, 0.906, 0.115, 0.42, 30.8, "ablation_tstart", "t_start=0.9"),

    # Ablation: patch_size (Places2 SDXL)
    ("SDXL", "Places2", "patch_size=2",                1.31, 21.24, 0.941, 0.076, 0.37, 38.4, "ablation_patch", "patch=2"),
    ("SDXL", "Places2", "patch_size=8",                1.26, 21.72, 0.947, 0.075, 0.30, 40.0, "ablation_patch", "patch=8"),

    # Cross-domain (draft trained on X, evaluated on Y)
    ("SDXL", "FFHQ",    "Places2-trained draft on FFHQ", 1.38, 20.63, 0.907, 0.138, 0.41, 36.4, "cross_domain", "P2-draft → FFHQ"),
    ("SDXL", "Places2", "FFHQ-trained draft on Places2", 1.19, 23.25, 0.957, 0.061, 0.32, 42.4, "cross_domain", "FFHQ-draft → P2"),
]


def write_all_csv():
    out_csv = "/mnt/HDD_12TB/bam_ki/results/all_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "setup", "speedup", "PSNR",
                    "SSIM", "LPIPS", "accept_rate", "NFE_target",
                    "category", "label"])
        for row in ALL_RESULTS:
            w.writerow(row)
    print(f"saved: {out_csv}")
    return out_csv


def write_ablation_csv():
    """Ablation 결과만 추출, paper table 형식으로."""
    out_csv = "/mnt/HDD_12TB/bam_ki/results/ablation_table.csv"
    rows = [r for r in ALL_RESULTS if r[9] != "main"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "setup", "model", "dataset",
                    "speedup", "PSNR", "SSIM", "LPIPS"])
        for r in rows:
            w.writerow([r[9], r[2], r[0], r[1], r[3], r[4], r[5], r[6]])
    print(f"saved: {out_csv}")
    return out_csv


def plot_pareto():
    """Pareto curve plot (4 subplot)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ----- Subplot (a): Main Pareto LPIPS vs Speedup -----
    ax = axes[0, 0]
    color_map = {
        ("SD2", "Places2"):  "tab:blue",
        ("SDXL", "Places2"): "tab:orange",
        ("SDXL", "FFHQ"):    "tab:green",
    }
    marker_map = {"default": "o", "mid": "^", "strict": "s"}

    for row in ALL_RESULTS:
        if row[9] != "main":
            continue
        model, ds, setup, sp, _, _, lp, _, _, _, lab = row
        # tolerance level from setup string
        if "default" in setup: marker = "o"
        elif "mid" in setup:    marker = "^"
        elif "strict" in setup: marker = "s"
        else:                   marker = "x"
        color = color_map.get((model, ds), "gray")
        ax.scatter(sp, lp, c=color, marker=marker, s=160,
                   edgecolors="black", linewidth=1.2, zorder=3)
        ax.annotate(lab, (sp, lp), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Speedup (×)", fontsize=11)
    ax.set_ylabel("LPIPS (lower = better)", fontsize=11)
    ax.set_title("(a) Main Pareto: Speedup vs LPIPS",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.95, 1.75)
    # Color legend
    from matplotlib.patches import Patch
    color_legend = [
        Patch(facecolor=color_map[("SD2", "Places2")],   label="SD2 Places2"),
        Patch(facecolor=color_map[("SDXL", "Places2")],  label="SDXL Places2"),
        Patch(facecolor=color_map[("SDXL", "FFHQ")],     label="SDXL FFHQ"),
    ]
    from matplotlib.lines import Line2D
    shape_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, markeredgecolor="black", label="default tol"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=10, markeredgecolor="black", label="mid tol"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=10, markeredgecolor="black", label="strict tol"),
    ]
    leg1 = ax.legend(handles=color_legend, loc="upper left",
                     fontsize=8, framealpha=0.9, title="Dataset")
    ax.add_artist(leg1)
    ax.legend(handles=shape_legend, loc="lower right",
              fontsize=8, framealpha=0.9, title="Tolerance")

    # ----- Subplot (b): K sweep -----
    ax = axes[0, 1]
    for ds, color in [("Places2", "tab:orange"), ("FFHQ", "tab:green")]:
        ks, sps, lps = [], [], []
        for r in ALL_RESULTS:
            if r[1] != ds: continue
            if r[9] == "ablation_K" or (r[9] == "main" and "default" in r[2]):
                # parse K
                if "K=1" in r[2]: k = 1
                elif "K=5" in r[2]: k = 5
                elif "K=3" in r[2] or "default" in r[2]: k = 3
                else: continue
                if r[0] != "SDXL": continue
                ks.append(k)
                sps.append(r[3])
                lps.append(r[6])
        if ks:
            # sort by K
            order = sorted(range(len(ks)), key=lambda i: ks[i])
            ks_s = [ks[i] for i in order]
            sps_s = [sps[i] for i in order]
            lps_s = [lps[i] for i in order]
            ax.plot(sps_s, lps_s, "-", color=color, alpha=0.5, linewidth=1.5)
            ax.scatter(sps_s, lps_s, c=color, s=140, edgecolors="black",
                       linewidth=1, zorder=3, label=f"SDXL {ds}")
            for k, sp, lp in zip(ks_s, sps_s, lps_s):
                ax.annotate(f"K={k}", (sp, lp), fontsize=8,
                            xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Speedup (×)", fontsize=11)
    ax.set_ylabel("LPIPS (lower = better)", fontsize=11)
    ax.set_title("(b) K sweep (K=1/3/5) at default tolerance",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # ----- Subplot (c): Ablation bar chart -----
    ax = axes[1, 0]
    ablations = []
    for r in ALL_RESULTS:
        if r[0] == "SDXL" and r[1] == "Places2":
            if r[2].startswith("default") and r[9] == "main":
                ablations.append(("FreqSpec (full)", r[3], r[6]))
            elif r[9] == "ablation_saliency":
                ablations.append(("no saliency", r[3], r[6]))
            elif r[9] == "ablation_boundary":
                ablations.append(("no boundary", r[3], r[6]))
    if ablations:
        names = [a[0] for a in ablations]
        sps   = [a[1] for a in ablations]
        lps   = [a[2] for a in ablations]
        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w/2, sps, w, label="Speedup (×)",
                       color="tab:orange", edgecolor="black")
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + w/2, lps, w, label="LPIPS",
                        color="tab:red", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Speedup (×)", color="tab:orange", fontsize=11)
        ax2.set_ylabel("LPIPS (lower better)", color="tab:red", fontsize=11)
        ax.set_title("(c) Ablation: SDXL Places2 default",
                     fontsize=12, fontweight="bold")
        for b in bars1:
            ax.annotate(f"{b.get_height():.2f}",
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
        for b in bars2:
            ax2.annotate(f"{b.get_height():.3f}",
                         (b.get_x() + b.get_width()/2, b.get_height()),
                         ha="center", va="bottom", fontsize=9)

    # ----- Subplot (d): Hyperparameter sensitivity -----
    ax = axes[1, 1]
    # t_spec_start sweep
    t_data = []
    for r in ALL_RESULTS:
        if r[0] == "SDXL" and r[1] == "Places2":
            if r[9] == "ablation_tstart":
                ts = float(r[2].split("=")[1])
                t_data.append((ts, r[3], r[6]))
            elif r[9] == "main" and "default" in r[2]:
                t_data.append((0.7, r[3], r[6]))
    t_data.sort()
    if t_data:
        ts = [d[0] for d in t_data]
        sps = [d[1] for d in t_data]
        lps = [d[2] for d in t_data]
        ax.plot(ts, sps, "o-", color="tab:orange", label="Speedup",
                linewidth=2, markersize=10, markeredgecolor="black")
        ax2 = ax.twinx()
        ax2.plot(ts, lps, "s-", color="tab:red", label="LPIPS",
                 linewidth=2, markersize=10, markeredgecolor="black")
        ax.set_xlabel("t_spec_start (Phase 1 stabilization end)", fontsize=11)
        ax.set_ylabel("Speedup (×)", color="tab:orange", fontsize=11)
        ax2.set_ylabel("LPIPS (lower better)", color="tab:red", fontsize=11)
        ax.set_title("(d) Stabilization length sensitivity",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # annotate sweet spot
        ax.axvline(0.7, color="green", linestyle="--", alpha=0.5)
        ax.annotate("default (sweet spot)",
                    (0.7, max(sps) * 0.7),
                    fontsize=9, color="green",
                    rotation=90, va="bottom")

    fig.suptitle("FreqSpec-Inpaint: Comprehensive Results",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    out_png = "/mnt/HDD_12TB/bam_ki/results/pareto_curve_v2.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_png.replace('.png', '.pdf')}")


def print_summary_tables():
    """Console에 paper-ready 표 출력."""
    print("\n" + "=" * 80)
    print("MAIN RESULTS (Pareto frontier points)")
    print("=" * 80)
    print(f"{'Setup':<35} {'Speedup':>8} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>7}")
    print("-" * 80)
    for r in ALL_RESULTS:
        if r[9] == "main":
            label = f"{r[0]} {r[1]} ({r[2].split('(')[0].strip()})"
            print(f"{label:<35} {r[3]:>7.2f}x {r[4]:>7.2f} {r[5]:>6.3f} {r[6]:>7.4f}")

    print("\n" + "=" * 80)
    print("ABLATION RESULTS (vs SDXL Places2 default: 1.35× / 22.08 / 0.948 / 0.074)")
    print("=" * 80)
    print(f"{'Category':<22} {'Setup':<25} {'Speedup':>8} {'PSNR':>7} {'LPIPS':>7}")
    print("-" * 80)
    for r in ALL_RESULTS:
        if r[9] == "main": continue
        print(f"{r[9]:<22} {r[10]:<25} {r[3]:>7.2f}x {r[4]:>7.2f} {r[6]:>7.4f}")


if __name__ == "__main__":
    write_all_csv()
    write_ablation_csv()
    print_summary_tables()
    plot_pareto()
    print("\nDone.")
