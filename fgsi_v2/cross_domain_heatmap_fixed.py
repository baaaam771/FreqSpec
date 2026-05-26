#!/usr/bin/env python
"""
Cross-Domain Transferability Heatmap (Figure 3) — fixed version.

Fix: COCO → COCO native cell now uses DEFAULT tolerance values
(LPIPS 0.077, speedup 1.23×) to match Table 2 and ensure all cells
use consistent tolerance settings.

Previous (incorrect):
  COCO → COCO: 0.043 LPIPS, 1.07× speedup (these were STRICT tolerance)

Now (correct):
  COCO → COCO: 0.077 LPIPS, 1.23× speedup (default tolerance, same as others)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ====================================================================
# Data — all at DEFAULT tolerance (0.03, 0.3), n=20
# ====================================================================
datasets = ["Places2", "FFHQ", "COCO"]

# LPIPS matrix: rows=train, cols=eval
# Lower is better
lpips_matrix = np.array([
    # eval: Places2,  FFHQ,    COCO
    [    0.074,    0.153,   np.nan],  # train: Places2
    [    0.082,    0.085,   np.nan],  # train: FFHQ
    [    0.073,    0.126,   0.077],   # train: COCO  ← FIXED: 0.043 → 0.077
])

# Speedup matrix: rows=train, cols=eval
# Higher is better
speedup_matrix = np.array([
    # eval: Places2,  FFHQ,    COCO
    [   1.35,     1.42,    np.nan],   # train: Places2
    [   1.31,     1.62,    np.nan],   # train: FFHQ
    [   1.35,     1.62,    1.23],     # train: COCO  ← FIXED: 1.07 → 1.23
])

# Native (in-distribution) cells marked for special annotation
native_cells = [(0, 0), (1, 1), (2, 2)]


# ====================================================================
# Figure setup
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))


# ====================================================================
# Panel (a) — LPIPS Heatmap (lower = better, hot colormap reversed)
# ====================================================================
ax = axes[0]
# Use a "RdYlGn_r" (reversed) so green=low (good), red=high (bad)
cmap_lpips = "RdYlGn_r"

# Mask NaN values (the COCO eval column for Places2/FFHQ rows)
masked_lpips = np.ma.masked_invalid(lpips_matrix)

im1 = ax.imshow(masked_lpips, cmap=cmap_lpips,
                vmin=0.05, vmax=0.16, aspect="equal")

# Annotate each cell
for i in range(3):
    for j in range(3):
        val = lpips_matrix[i, j]
        if np.isnan(val):
            ax.text(j, i, "—", ha="center", va="center",
                    fontsize=18, color="gray", fontweight="bold")
            continue
        # Native cells: extra label
        is_native = (i, j) in native_cells
        # Text color: contrast based on background
        text_color = "white" if val > 0.10 else "black"
        if is_native:
            ax.text(j, i, f"{val:.3f}\n(native)",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)
        else:
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

# Axis labels and ticks
ax.set_xticks(range(3))
ax.set_xticklabels(datasets, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(datasets, fontsize=11)
ax.set_xlabel("Evaluation Dataset", fontsize=12, fontweight="bold")
ax.set_ylabel("Training Dataset", fontsize=12, fontweight="bold")
ax.set_title("(a) LPIPS Heatmap (lower = better)",
             fontsize=12, fontweight="bold", pad=10)

# Colorbar
cbar1 = plt.colorbar(im1, ax=ax, shrink=0.85, pad=0.02)
cbar1.set_label("LPIPS", fontsize=10)

# Grid (subtle)
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)


# ====================================================================
# Panel (b) — Speedup Heatmap (higher = better, green=high)
# ====================================================================
ax = axes[1]
cmap_speedup = "RdYlGn"  # green=high (good), red=low (bad)

masked_speedup = np.ma.masked_invalid(speedup_matrix)

im2 = ax.imshow(masked_speedup, cmap=cmap_speedup,
                vmin=1.0, vmax=1.7, aspect="equal")

# Annotate
for i in range(3):
    for j in range(3):
        val = speedup_matrix[i, j]
        if np.isnan(val):
            ax.text(j, i, "—", ha="center", va="center",
                    fontsize=18, color="gray", fontweight="bold")
            continue
        is_native = (i, j) in native_cells
        text_color = "white" if val < 1.20 else "black"
        if is_native:
            ax.text(j, i, f"{val:.2f}$\\times$\n(native)",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)
        else:
            ax.text(j, i, f"{val:.2f}$\\times$",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

ax.set_xticks(range(3))
ax.set_xticklabels(datasets, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(datasets, fontsize=11)
ax.set_xlabel("Evaluation Dataset", fontsize=12, fontweight="bold")
ax.set_ylabel("Training Dataset", fontsize=12, fontweight="bold")
ax.set_title("(b) Speedup Heatmap (higher = better)",
             fontsize=12, fontweight="bold", pad=10)

cbar2 = plt.colorbar(im2, ax=ax, shrink=0.85, pad=0.02)
cbar2.set_label(r"Speedup ($\times$)", fontsize=10)

# Grid
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)


# ====================================================================
# Overall title and layout
# ====================================================================
fig.suptitle(
    "Cross-Domain Transferability (n=20, default tolerance)",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()

# Save
out_png = "/mnt/HDD_12TB/bam_ki/results/cross_domain_heatmap.png"
out_pdf = "/mnt/HDD_12TB/bam_ki/results/cross_domain_heatmap.pdf"
plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close()

print(f"saved: {out_png}")
print(f"saved: {out_pdf}")

# Verify the fix
print("\n=== Cross-Domain Matrix (default tolerance, n=20) ===")
print(f"{'':12} {'Places2':>12} {'FFHQ':>12} {'COCO':>12}")
for i, name in enumerate(datasets):
    row_str = f"{name:12}"
    for j in range(3):
        lp = lpips_matrix[i, j]
        sp = speedup_matrix[i, j]
        if np.isnan(lp):
            row_str += f" {'—':>12}"
        else:
            marker = " (native)" if (i, j) in native_cells else ""
            row_str += f" {sp:.2f}x, {lp:.3f}{marker}"
    print(row_str)
