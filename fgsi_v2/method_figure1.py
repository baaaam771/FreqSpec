#!/usr/bin/env python
"""
Method overview figure for FreqSpec-Inpaint paper.

Generates Figure 1: 3-panel method overview diagram.
  (a) Saliency Computation
  (b) Speculative Loop (Phase 1 + Phase 2 + Adaptive K-switching)
  (c) Per-Patch Acceptance Criterion

Saves to /mnt/HDD_12TB/bam_ki/results/method_figure1.png/pdf
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np


# Colors (consistent with paper)
C_DRAFT = "#FF8C00"      # Orange — draft model
C_TARGET = "#1F77B4"     # Blue — target model
C_PHASE1 = "#88C0D0"     # Light blue — Phase 1
C_PHASE2_K = "#A3BE8C"   # Green — Phase 2 spec-K
C_PHASE2_1 = "#EBCB8B"   # Yellow — Phase 2 spec-1
C_ACCEPT = "#A3BE8C"     # Green — accept
C_REJECT = "#BF616A"     # Red — reject
C_BG = "#ECEFF4"         # Light gray bg
C_TEXT = "#2E3440"       # Dark text


def panel_a_saliency(ax):
    """(a) Saliency Computation"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Frequency-Guided Saliency",
                 fontsize=12, fontweight="bold", pad=10)

    # Input: noisy latent z_t with mask
    np.random.seed(42)
    # Simulate latent (random texture)
    z_data = np.random.randn(32, 32) * 0.3
    # Add some structure
    for cy, cx, r in [(10, 10, 4), (20, 22, 5)]:
        y, x = np.ogrid[:32, :32]
        z_data[(y-cy)**2 + (x-cx)**2 <= r**2] += 1.5
    # Display as inset
    inset1 = ax.inset_axes([0.05, 0.6, 0.25, 0.25])
    inset1.imshow(z_data, cmap="gray")
    inset1.set_xticks([])
    inset1.set_yticks([])
    inset1.set_title("$z_t$", fontsize=10)
    for spine in inset1.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    # Mask
    mask = np.zeros((32, 32))
    mask[8:24, 8:24] = 1  # central box
    inset2 = ax.inset_axes([0.05, 0.25, 0.25, 0.25])
    inset2.imshow(mask, cmap="gray", vmin=0, vmax=1)
    inset2.set_xticks([])
    inset2.set_yticks([])
    inset2.set_title("Mask $M$", fontsize=10)
    for spine in inset2.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    # DWT block
    dwt_box = FancyBboxPatch((4, 7), 2.2, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=C_BG, edgecolor=C_TEXT, linewidth=1.5)
    ax.add_patch(dwt_box)
    ax.text(5.1, 7.6, "DWT\n(Haar)", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Boundary indicator block
    bnd_box = FancyBboxPatch((4, 3.5), 2.2, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=C_BG, edgecolor=C_TEXT, linewidth=1.5)
    ax.add_patch(bnd_box)
    ax.text(5.1, 4.1, "Boundary\n$B_{ind}$", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # Arrows: z_t → DWT
    arr1 = FancyArrowPatch((2.7, 8), (4, 7.6),
                            arrowstyle="->", mutation_scale=15,
                            color=C_TEXT, linewidth=1.5)
    ax.add_patch(arr1)
    # Mask → Boundary
    arr2 = FancyArrowPatch((2.7, 4), (4, 4.1),
                            arrowstyle="->", mutation_scale=15,
                            color=C_TEXT, linewidth=1.5)
    ax.add_patch(arr2)

    # Combine (+) node
    ax.text(7.3, 5.85, "$+$", ha="center", va="center",
            fontsize=22, fontweight="bold", color=C_TEXT)
    arr3 = FancyArrowPatch((6.2, 7.6), (7.0, 6.2),
                            arrowstyle="->", mutation_scale=15,
                            color=C_TEXT, linewidth=1.5)
    arr4 = FancyArrowPatch((6.2, 4.1), (7.0, 5.5),
                            arrowstyle="->", mutation_scale=15,
                            color=C_TEXT, linewidth=1.5)
    ax.add_patch(arr3)
    ax.add_patch(arr4)

    # Output saliency map A (heatmap visualization)
    # Combined: high frequency + boundary
    sal = np.zeros((32, 32))
    sal[10:14, 8:14] = 0.9   # high-freq region
    sal[18:24, 20:26] = 0.7
    # boundary
    sal[7:9, 8:24] = 0.8
    sal[23:25, 8:24] = 0.8
    sal[8:24, 7:9] = 0.8
    sal[8:24, 23:25] = 0.8
    # Smooth slightly
    from scipy.ndimage import gaussian_filter
    try:
        sal = gaussian_filter(sal, sigma=1.0)
    except ImportError:
        pass

    inset3 = ax.inset_axes([0.65, 0.4, 0.32, 0.32])
    inset3.imshow(sal, cmap="hot", vmin=0, vmax=1)
    inset3.set_xticks([])
    inset3.set_yticks([])
    inset3.set_title("Saliency $A$", fontsize=10, fontweight="bold")
    for spine in inset3.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    # Caption below
    ax.text(5, 0.5,
            r"$A = \text{norm}(A_w + \lambda_b \cdot B_{ind})$",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_TEXT, linewidth=1))


def panel_b_speculative(ax):
    """(b) Speculative Loop"""
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("(b) Two-Phase Speculative Inference",
                 fontsize=12, fontweight="bold", pad=10)

    # Timeline horizontal bar
    timeline_y = 7.5
    ax.add_patch(Rectangle((1, timeline_y - 0.4), 12, 0.8,
                            facecolor=C_BG, edgecolor=C_TEXT, linewidth=1.5))
    # Phase 1: target only (left 30%)
    ax.add_patch(Rectangle((1, timeline_y - 0.4), 3.6, 0.8,
                            facecolor=C_PHASE1, edgecolor=C_TEXT, linewidth=1.5))
    # Phase 2 spec-K (middle ~50%)
    ax.add_patch(Rectangle((4.6, timeline_y - 0.4), 6, 0.8,
                            facecolor=C_PHASE2_K, edgecolor=C_TEXT, linewidth=1.5))
    # Phase 2 spec-1 (right ~20%)
    ax.add_patch(Rectangle((10.6, timeline_y - 0.4), 2.4, 0.8,
                            facecolor=C_PHASE2_1, edgecolor=C_TEXT, linewidth=1.5))

    # Labels on phases (with white background to overlay any line crossing)
    ax.text(2.8, timeline_y, "Phase 1: target only", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C_PHASE1,
                      edgecolor="none", alpha=1.0))
    ax.text(7.6, timeline_y, "Phase 2: spec-$K$", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C_PHASE2_K,
                      edgecolor="none", alpha=1.0))
    ax.text(11.8, timeline_y, "spec-1", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C_PHASE2_1,
                      edgecolor="none", alpha=1.0))

    # Timestep labels (t direction) — placed above the timeline
    ax.annotate("", xy=(13.3, timeline_y + 1.3), xytext=(0.7, timeline_y + 1.3),
                arrowprops=dict(arrowstyle="<->", color=C_TEXT, lw=1.5))
    ax.text(0.5, timeline_y + 1.6, "$t = T$ (noise)", ha="left", va="bottom",
            fontsize=9, style="italic")
    ax.text(13.5, timeline_y + 1.6, "$t = 0$ (clean)", ha="right", va="bottom",
            fontsize=9, style="italic")

    # Switch indicator: t_spec — short line just above & below timeline
    ax.plot([4.6, 4.6], [timeline_y + 0.4, timeline_y + 1.0],
            color=C_TEXT, linestyle="--", linewidth=2)
    ax.plot([4.6, 4.6], [timeline_y - 0.8, timeline_y - 0.4],
            color=C_TEXT, linestyle="--", linewidth=2)
    ax.text(4.6, timeline_y - 1.0, "$t_{spec}$", ha="center", va="top",
            fontsize=10, fontweight="bold", color=C_TEXT)

    # Switch indicator: spec-K → spec-1
    ax.plot([10.6, 10.6], [timeline_y + 0.4, timeline_y + 1.0],
            color=C_TEXT, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.plot([10.6, 10.6], [timeline_y - 0.8, timeline_y - 0.4],
            color=C_TEXT, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(10.6, timeline_y - 1.0, "accept rate\n< 0.5",
            ha="center", va="top", fontsize=8,
            style="italic", color=C_TEXT)

    # ============= Lower part: zoom on one Phase-2 step =============
    # Draw a "zoom" indicator
    zoom_top_y = timeline_y - 2.0
    ax.annotate("", xy=(7.6, zoom_top_y), xytext=(7.6, timeline_y - 0.7),
                arrowprops=dict(arrowstyle="->", color=C_TEXT,
                                lw=1.5, linestyle=":"))
    ax.text(8.0, zoom_top_y + 0.3, "zoom: one step",
            ha="left", va="center", fontsize=8, style="italic")

    # Block: z_t (input)
    ax.add_patch(Rectangle((1, 2.5), 1.5, 1.2,
                            facecolor="white", edgecolor=C_TEXT, linewidth=1.5))
    ax.text(1.75, 3.1, "$z_t$", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Block: Draft (parallel K steps)
    ax.add_patch(FancyBboxPatch((3.5, 3.7), 2.3, 1.1,
                                 boxstyle="round,pad=0.05",
                                 facecolor=C_DRAFT, edgecolor=C_TEXT,
                                 linewidth=1.5, alpha=0.8))
    ax.text(4.65, 4.25, "Draft $\\epsilon_\\phi^{drf}$\n(× $K$ steps)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Block: Target (single step)
    ax.add_patch(FancyBboxPatch((3.5, 1.7), 2.3, 1.1,
                                 boxstyle="round,pad=0.05",
                                 facecolor=C_TARGET, edgecolor=C_TEXT,
                                 linewidth=1.5, alpha=0.7))
    ax.text(4.65, 2.25, "Target $\\epsilon_\\theta^{tgt}$\n(reference)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white")

    # Arrows from z_t to both
    ax.add_patch(FancyArrowPatch((2.55, 3.4), (3.5, 4.25),
                                  arrowstyle="->", mutation_scale=15,
                                  color=C_TEXT, linewidth=1.5))
    ax.add_patch(FancyArrowPatch((2.55, 2.8), (3.5, 2.25),
                                  arrowstyle="->", mutation_scale=15,
                                  color=C_TEXT, linewidth=1.5))

    # Acceptance diamond (Eq. 14)
    diamond_x, diamond_y = 7.5, 3.1
    diamond = mpatches.Polygon([(diamond_x - 1.1, diamond_y),
                                 (diamond_x, diamond_y + 1.0),
                                 (diamond_x + 1.1, diamond_y),
                                 (diamond_x, diamond_y - 1.0)],
                                facecolor=C_BG, edgecolor=C_TEXT, linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y + 0.05,
            "$\\|\\epsilon_{drf} - \\epsilon_{tgt}\\|$",
            ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(diamond_x, diamond_y - 0.35, "$< \\tau(A)$ ?",
            ha="center", va="center", fontsize=8, fontweight="bold")

    # Arrows draft → acceptance
    ax.add_patch(FancyArrowPatch((5.85, 4.25), (6.4, 3.5),
                                  arrowstyle="->", mutation_scale=15,
                                  color=C_DRAFT, linewidth=1.5))
    ax.add_patch(FancyArrowPatch((5.85, 2.25), (6.4, 2.7),
                                  arrowstyle="->", mutation_scale=15,
                                  color=C_TARGET, linewidth=1.5))

    # Two outputs: accept (skip K-1 steps) or reject (use target)
    ax.add_patch(FancyBboxPatch((9.5, 4.0), 3.5, 1.0,
                                 boxstyle="round,pad=0.05",
                                 facecolor=C_ACCEPT, edgecolor=C_TEXT,
                                 linewidth=1.5, alpha=0.6))
    ax.text(11.25, 4.5, "Accept ✓",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(11.25, 4.10, "skip $K-1$ target NFEs",
            ha="center", va="center", fontsize=8, style="italic")

    ax.add_patch(FancyBboxPatch((9.5, 1.8), 3.5, 1.0,
                                 boxstyle="round,pad=0.05",
                                 facecolor=C_REJECT, edgecolor=C_TEXT,
                                 linewidth=1.5, alpha=0.6))
    ax.text(11.25, 2.30, "Reject ✗",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white")
    ax.text(11.25, 1.95, "blend with target",
            ha="center", va="center", fontsize=8, style="italic",
            color="white")

    # Arrows from acceptance
    ax.add_patch(FancyArrowPatch((8.6, 3.5), (9.5, 4.5),
                                  arrowstyle="->", mutation_scale=15,
                                  color=C_ACCEPT, linewidth=1.5))
    ax.add_patch(FancyArrowPatch((8.6, 2.7), (9.5, 2.3),
                                  arrowstyle="->", mutation_scale=15,
                                  color=C_REJECT, linewidth=1.5))


def panel_c_acceptance(ax):
    """(c) Per-Patch Acceptance Criterion"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) Per-Patch Acceptance",
                 fontsize=12, fontweight="bold", pad=10)

    # 6x6 patch grid
    grid_size = 6
    cell_size = 1.0
    grid_x, grid_y = 1.0, 3.0

    # Per-patch (saliency, delta) — design example
    np.random.seed(42)
    saliency_grid = np.zeros((grid_size, grid_size))
    saliency_grid[1:3, 1:3] = 0.8   # high-freq region
    saliency_grid[3:5, 3:5] = 0.7
    saliency_grid[0, :] = 0.5        # mask boundary top
    saliency_grid[5, :] = 0.5        # mask boundary bottom
    saliency_grid[:, 0] = 0.5
    saliency_grid[:, 5] = 0.5
    # Smooth
    from scipy.ndimage import gaussian_filter
    try:
        saliency_grid = gaussian_filter(saliency_grid, sigma=0.5)
    except ImportError:
        pass

    # delta values: roughly inverse correlation (high saliency = high delta)
    delta_grid = saliency_grid * 0.3 + np.random.uniform(0, 0.1, (grid_size, grid_size))

    # tau threshold
    tau_l, tau_h = 0.03, 0.3
    tau_grid = tau_h * saliency_grid + tau_l * (1 - saliency_grid)

    # Accept where delta < tau
    accept_grid = delta_grid < tau_grid

    # Draw cells
    for i in range(grid_size):
        for j in range(grid_size):
            x = grid_x + j * cell_size
            y = grid_y + (grid_size - 1 - i) * cell_size
            color = C_ACCEPT if accept_grid[i, j] else C_REJECT
            alpha = 0.7 if accept_grid[i, j] else 0.6
            ax.add_patch(Rectangle((x, y), cell_size, cell_size,
                                    facecolor=color, edgecolor=C_TEXT,
                                    linewidth=1, alpha=alpha))
            # Annotation: small symbol
            symbol = "✓" if accept_grid[i, j] else "✗"
            txt_color = "black" if accept_grid[i, j] else "white"
            ax.text(x + 0.5, y + 0.5, symbol, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)

    # Legend below grid
    ax.add_patch(Rectangle((1.5, 1.2), 0.6, 0.6,
                            facecolor=C_ACCEPT, edgecolor=C_TEXT,
                            linewidth=1, alpha=0.7))
    ax.text(2.3, 1.5, "Use draft", ha="left", va="center",
            fontsize=9, fontweight="bold")

    ax.add_patch(Rectangle((4.5, 1.2), 0.6, 0.6,
                            facecolor=C_REJECT, edgecolor=C_TEXT,
                            linewidth=1, alpha=0.6))
    ax.text(5.3, 1.5, "Use target", ha="left", va="center",
            fontsize=9, fontweight="bold")

    # Equation
    ax.text(5, 0.4,
            r"$\tau(i,j) = \tau_h \cdot A(i,j) + \tau_l \cdot (1 - A(i,j))$",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_TEXT, linewidth=1))

    # Top: explain saliency-tolerance relation
    ax.text(5, 9.5,
            "Low saliency $\\Rightarrow$ loose $\\tau$ $\\Rightarrow$ accept easily\nHigh saliency $\\Rightarrow$ strict $\\tau$ $\\Rightarrow$ reject often",
            ha="center", va="top", fontsize=8, style="italic")


def main():
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.08)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    panel_a_saliency(ax_a)
    panel_b_speculative(ax_b)
    panel_c_acceptance(ax_c)

    fig.suptitle("FreqSpec-Inpaint: Method Overview",
                 fontsize=15, fontweight="bold", y=1.02)

    out_png = "/mnt/HDD_12TB/bam_ki/results/method_figure1.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"saved: {out_png}")
    print(f"saved: {out_png.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
