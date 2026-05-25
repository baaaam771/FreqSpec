#!/usr/bin/env python
"""
Qualitative Figure for FreqSpec-Inpaint paper (Figure 5).

Shows visual evidence for 3 key findings:
  Row 1: Captions vs Fixed prompt (COCO, same image)
  Row 2: Cross-domain failure (COCO draft on FFHQ face)
  Row 3: Tolerance Pareto (default vs strict on natural scene)

IMPORTANT: Update RESULT_ROOT and image paths to match actual filesystem.
Run after verifying paths exist with:
  ls /mnt/HDD_12TB/bam_ki/results/sdxl_coco_eval_captions/
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np


RESULT_ROOT = "/mnt/HDD_12TB/bam_ki/results"


# ====================================================================
# Image loading helpers
# ====================================================================
def load_img(path, fallback_size=(256, 256)):
    """Load image with fallback to gray placeholder if missing."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except (FileNotFoundError, IOError):
        print(f"  ⚠ Missing: {path}")
        # gray placeholder
        gray = np.ones((*fallback_size, 3), dtype=np.uint8) * 128
        return gray


def show_img(ax, path, title=None, ylabel=None, title_color="black"):
    """Display image with optional title and row label."""
    img = load_img(path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=title_color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold",
                      rotation=90, labelpad=10)


def add_metric_box(ax, text, position="bottom"):
    """Add a metric annotation box on top of an image."""
    if position == "bottom":
        ax.text(0.5, 0.05, text, transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.7,
                          edgecolor="white", linewidth=0.5))
    else:  # top
        ax.text(0.5, 0.95, text, transform=ax.transAxes,
                ha="center", va="top",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.7,
                          edgecolor="white", linewidth=0.5))


# ====================================================================
# Build figure
# ====================================================================
def main():
    # ACTUAL paths discovered from filesystem:
    # results/<eval_name>/img_NNN/
    # Files: input.png, mask.png, masked.png, out_baseline.png, out_fgsr.png, saliency.png
    paths = {
        # ROW 1: Caption vs Fixed (same image, different prompt strategy)
        # img_003 was the giraffe case where fixed-prompt added a girl
        "r1_input":         f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_003/masked.png",
        "r1_baseline":      f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_003/out_baseline.png",
        "r1_cap_freqspec":  f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_003/out_fgsr.png",
        "r1_fix_freqspec":  f"{RESULT_ROOT}/sdxl_coco_eval_fixed/img_003/out_fgsr.png",

        # ROW 2: Cross-domain failure (COCO draft on FFHQ vs FFHQ-native draft)
        "r2_input":     f"{RESULT_ROOT}/cross_cocodraft_on_ffhq_n20/img_001/masked.png",
        "r2_baseline":  f"{RESULT_ROOT}/cross_cocodraft_on_ffhq_n20/img_001/out_baseline.png",
        "r2_native":    f"{RESULT_ROOT}/sdxl_ffhq_eval/img_001/out_fgsr.png",  # FFHQ-native
        "r2_cross":     f"{RESULT_ROOT}/cross_cocodraft_on_ffhq_n20/img_001/out_fgsr.png",  # COCO-cross

        # ROW 3: Tolerance trade-off (default vs strict on Places2)
        "r3_input":     f"{RESULT_ROOT}/sdxl_places2_seed200/img_001/masked.png",
        "r3_baseline":  f"{RESULT_ROOT}/sdxl_places2_seed200/img_001/out_baseline.png",
        "r3_default":   f"{RESULT_ROOT}/sdxl_places2_seed200/img_001/out_fgsr.png",
        "r3_strict":    f"{RESULT_ROOT}/sdxl_places2_strict/img_001/out_fgsr.png",
    }

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.18, wspace=0.05,
                  left=0.08, right=0.98, top=0.93, bottom=0.05)

    # ============================================================
    # ROW 1: Caption vs Fixed Prompt
    # ============================================================
    ax = fig.add_subplot(gs[0, 0])
    show_img(ax, paths["r1_input"],
             title="Input (masked)",
             ylabel="(a) Captions vs Fixed Prompt\n(COCO)")

    ax = fig.add_subplot(gs[0, 1])
    show_img(ax, paths["r1_baseline"], title="Baseline (target only)")
    add_metric_box(ax, "PSNR 24.0\nLPIPS 0.07")

    ax = fig.add_subplot(gs[0, 2])
    show_img(ax, paths["r1_cap_freqspec"],
             title="FreqSpec (captions)",
             title_color="darkgreen")
    add_metric_box(ax, "PSNR 23.4\nLPIPS 0.08\n1.47×")

    ax = fig.add_subplot(gs[0, 3])
    show_img(ax, paths["r1_fix_freqspec"],
             title="FreqSpec (fixed prompt)",
             title_color="darkred")
    add_metric_box(ax, "PSNR 22.1\nLPIPS 0.10\n1.56×")

    # ============================================================
    # ROW 2: Cross-domain Failure (Face)
    # ============================================================
    ax = fig.add_subplot(gs[1, 0])
    show_img(ax, paths["r2_input"],
             title="Input (masked face)",
             ylabel="(b) Cross-Domain Failure\n(FFHQ Eval)")

    ax = fig.add_subplot(gs[1, 1])
    show_img(ax, paths["r2_baseline"], title="Baseline (target only)")
    add_metric_box(ax, "PSNR 22.4\nLPIPS 0.05")

    ax = fig.add_subplot(gs[1, 2])
    show_img(ax, paths["r2_native"],
             title="FreqSpec (FFHQ-native draft)",
             title_color="darkgreen")
    add_metric_box(ax, "PSNR 22.4\nLPIPS 0.09\n1.62×")

    ax = fig.add_subplot(gs[1, 3])
    show_img(ax, paths["r2_cross"],
             title="FreqSpec (COCO-cross draft)",
             title_color="darkred")
    add_metric_box(ax, "PSNR 20.8\nLPIPS 0.13\n1.62×")

    # ============================================================
    # ROW 3: Tolerance Pareto
    # ============================================================
    ax = fig.add_subplot(gs[2, 0])
    show_img(ax, paths["r3_input"],
             title="Input (masked)",
             ylabel="(c) Tolerance Trade-off\n(Places2)")

    ax = fig.add_subplot(gs[2, 1])
    show_img(ax, paths["r3_baseline"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")

    ax = fig.add_subplot(gs[2, 2])
    show_img(ax, paths["r3_default"],
             title="FreqSpec (default τ)",
             title_color="darkblue")
    add_metric_box(ax, "PSNR 22.1\nLPIPS 0.07\n1.35×")

    ax = fig.add_subplot(gs[2, 3])
    show_img(ax, paths["r3_strict"],
             title="FreqSpec (strict τ)",
             title_color="darkblue")
    add_metric_box(ax, "PSNR 25.9\nLPIPS 0.03\n1.05×")

    # ============================================================
    # Overall title
    # ============================================================
    fig.suptitle(
        "Figure 5: Qualitative Results — Three Failure/Success Modes",
        fontsize=14, fontweight="bold", y=0.97
    )

    out_png = "/mnt/HDD_12TB/bam_ki/results/qualitative_figure5.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"saved: {out_png}")
    print(f"saved: {out_png.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()