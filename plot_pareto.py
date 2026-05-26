#!/usr/bin/env python
"""Pareto curve plot for FreqSpec-Inpaint paper.

Run on bam_ki@mrlab-pro6000:
    python plot_pareto.py
"""
import matplotlib.pyplot as plt
import numpy as np

# Measured results
data = [
    # (label, speedup, LPIPS, PSNR, color, marker)
    ("SD-Inpaint Places2\n(default)",     1.28, 0.064, 22.83, "tab:blue",   "o"),
    ("SDXL-Inpaint Places2\n(default)",   1.35, 0.074, 22.08, "tab:orange", "o"),
    ("SDXL-Inpaint Places2\n(strict)",    1.05, 0.029, 25.88, "tab:orange", "s"),
    ("SDXL-Inpaint FFHQ\n(default)",      1.62, 0.085, 22.44, "tab:green",  "o"),
    ("SDXL-Inpaint FFHQ\n(mid)",          1.36, 0.043, 25.23, "tab:green",  "^"),
    ("SDXL-Inpaint FFHQ\n(strict)",       1.12, 0.025, 27.58, "tab:green",  "s"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Speedup vs LPIPS
ax1 = axes[0]
for label, sp, lp, ps, color, marker in data:
    ax1.scatter(sp, lp, c=color, marker=marker, s=150,
                edgecolors="black", linewidth=1.2, zorder=3, label=label)

# Annotation arrow for FFHQ default (visible artifacts)
ax1.annotate("artifacts visible\nbut LPIPS comparable",
             xy=(1.62, 0.085), xytext=(1.40, 0.095),
             fontsize=8, color="darkred", style="italic",
             arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

ax1.set_xlabel("Speedup (×)", fontsize=11)
ax1.set_ylabel("LPIPS (lower is better)", fontsize=11)
ax1.set_title("(a) Speedup vs Perceptual Quality (LPIPS)",
              fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1.0, 1.7)
ax1.set_ylim(0.0, 0.11)
ax1.invert_yaxis()  # lower LPIPS = better, so put top of plot
ax1.legend(fontsize=7, loc="lower left", framealpha=0.9)

# Plot 2: Speedup vs PSNR
ax2 = axes[1]
for label, sp, lp, ps, color, marker in data:
    ax2.scatter(sp, ps, c=color, marker=marker, s=150,
                edgecolors="black", linewidth=1.2, zorder=3, label=label)

ax2.set_xlabel("Speedup (×)", fontsize=11)
ax2.set_ylabel("PSNR mask (dB, higher is better)", fontsize=11)
ax2.set_title("(b) Speedup vs PSNR",
              fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1.0, 1.7)
ax2.set_ylim(20, 30)
ax2.legend(fontsize=7, loc="upper right", framealpha=0.9)

# Tolerance shape legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="s", color="w", label="strict (0.01/0.1)",
           markerfacecolor="gray", markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker="^", color="w", label="mid (0.02/0.15)",
           markerfacecolor="gray", markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker="o", color="w", label="default (0.03/0.3)",
           markerfacecolor="gray", markersize=10, markeredgecolor="black"),
]
ax2.legend(handles=legend_elements, fontsize=9, loc="lower right",
           title="Tolerance", title_fontsize=10)

# Overall title
fig.suptitle("FreqSpec-Inpaint: Domain-Adaptive Quality-Speedup Trade-off",
             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()
out = "/mnt/HDD_12TB/bam_ki/results/pareto_curve.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"saved: {out.replace('.png', '.pdf')}")
