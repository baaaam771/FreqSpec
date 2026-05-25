#!/usr/bin/env python
"""
Qualitative Figure 5 — v4 with 5-column layout.

Each FreqSpec result is paired with its own baseline for direct comparison.

Layout:
  Col 1: Input (masked)
  Col 2: Baseline (setting A)
  Col 3: FreqSpec (setting A)
  Col 4: Baseline (setting B)
  Col 5: FreqSpec (setting B)

Rows:
  (a) Captions vs Fixed (COCO, img_010)
  (b) Default vs Strict tolerance on faces (FFHQ, img_001)
  (c) Default vs Strict tolerance on scenes (COCO, img_004)
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np


RESULT_ROOT = "/mnt/HDD_12TB/bam_ki/results"


def load_img(path, fallback_size=(256, 256)):
    """Load image with fallback to gray placeholder if missing."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except (FileNotFoundError, IOError):
        print(f"  ⚠ Missing: {path}")
        gray = np.ones((*fallback_size, 3), dtype=np.uint8) * 128
        return gray


def show_img(ax, path, title=None, ylabel=None, title_color="black"):
    img = load_img(path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=title_color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold",
                      rotation=90, labelpad=10)


def add_metric_box(ax, text):
    ax.text(0.5, 0.05, text, transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="black", alpha=0.7,
                      edgecolor="white", linewidth=0.5))


def add_setting_label(ax, text, color="black"):
    """Add a setting label above the column pair (Baseline-FreqSpec)."""
    ax.text(1.05, 1.08, text, transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.95,
                      edgecolor=color, linewidth=1.5))


def main():
    paths = {
        # ROW 1: Caption vs Fixed
        "r1_input":            f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_010/masked.png",
        "r1_baseline_cap":     f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_010/out_baseline.png",
        "r1_freqspec_cap":     f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_010/out_fgsr.png",
        "r1_baseline_fix":     f"{RESULT_ROOT}/sdxl_coco_eval_fixed/img_010/out_baseline.png",
        "r1_freqspec_fix":     f"{RESULT_ROOT}/sdxl_coco_eval_fixed/img_010/out_fgsr.png",

        # ROW 2: Default vs Strict tolerance on face (FFHQ)
        "r2_input":            f"{RESULT_ROOT}/sdxl_ffhq_eval/img_001/masked.png",
        "r2_baseline_def":     f"{RESULT_ROOT}/sdxl_ffhq_eval/img_001/out_baseline.png",
        "r2_freqspec_def":     f"{RESULT_ROOT}/sdxl_ffhq_eval/img_001/out_fgsr.png",
        "r2_baseline_strict":  f"{RESULT_ROOT}/sdxl_ffhq_strict/img_001/out_baseline.png",
        "r2_freqspec_strict":  f"{RESULT_ROOT}/sdxl_ffhq_strict/img_001/out_fgsr.png",

        # ROW 3: Default vs Strict tolerance on COCO scene
        "r3_input":            f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_004/masked.png",
        "r3_baseline_def":     f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_004/out_baseline.png",
        "r3_freqspec_def":     f"{RESULT_ROOT}/sdxl_coco_eval_captions/img_004/out_fgsr.png",
        "r3_baseline_strict":  f"{RESULT_ROOT}/sdxl_coco_strict/img_004/out_baseline.png",
        "r3_freqspec_strict":  f"{RESULT_ROOT}/sdxl_coco_strict/img_004/out_fgsr.png",
    }

    fig = plt.figure(figsize=(22, 14.5))
    gs = GridSpec(3, 5, figure=fig, hspace=0.40, wspace=0.04,
                  left=0.06, right=0.99, top=0.87, bottom=0.04)

    # ============================================================
    # ROW 1: Captions vs Fixed prompt
    # ============================================================
    ax = fig.add_subplot(gs[0, 0])
    show_img(ax, paths["r1_input"],
             title="Input (masked)",
             ylabel="(a) Captions vs Fixed Prompt\n(COCO)")

    # Setting A: Captions (cols 1-2 pair, label spans both)
    ax = fig.add_subplot(gs[0, 1])
    show_img(ax, paths["r1_baseline_cap"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")
    add_setting_label(ax, "Setting A: Per-image captions",
                      color="darkgreen")

    ax = fig.add_subplot(gs[0, 2])
    show_img(ax, paths["r1_freqspec_cap"], title="FreqSpec",
             title_color="darkgreen")
    add_metric_box(ax, "PSNR 20.0\nLPIPS 0.05\n1.19×")

    # Setting B: Fixed prompt (cols 3-4 pair)
    ax = fig.add_subplot(gs[0, 3])
    show_img(ax, paths["r1_baseline_fix"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")
    add_setting_label(ax, "Setting B: Fixed prompt ('a photograph')",
                      color="darkred")

    ax = fig.add_subplot(gs[0, 4])
    show_img(ax, paths["r1_freqspec_fix"], title="FreqSpec",
             title_color="darkred")
    add_metric_box(ax, "PSNR 19.9\nLPIPS 0.08\n1.32×")

    # ============================================================
    # ROW 2: Default vs Strict tolerance (Face / FFHQ)
    # ============================================================
    ax = fig.add_subplot(gs[1, 0])
    show_img(ax, paths["r2_input"],
             title="Input (masked face)",
             ylabel="(b) Face Inpainting Difficulty\n(FFHQ, native draft)")

    ax = fig.add_subplot(gs[1, 1])
    show_img(ax, paths["r2_baseline_def"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")
    add_setting_label(ax, "Setting A: Default τ=(0.03, 0.3)",
                      color="darkred")

    ax = fig.add_subplot(gs[1, 2])
    show_img(ax, paths["r2_freqspec_def"], title="FreqSpec",
             title_color="darkred")
    add_metric_box(ax, "PSNR 22.4\nLPIPS 0.085\n1.62×")

    ax = fig.add_subplot(gs[1, 3])
    show_img(ax, paths["r2_baseline_strict"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")
    add_setting_label(ax, "Setting B: Strict τ=(0.01, 0.1)",
                      color="darkgreen")

    ax = fig.add_subplot(gs[1, 4])
    show_img(ax, paths["r2_freqspec_strict"], title="FreqSpec",
             title_color="darkgreen")
    add_metric_box(ax, "PSNR 27.6\nLPIPS 0.025\n1.12×")

    # ============================================================
    # ROW 3: Default vs Strict tolerance (Scene / COCO)
    # ============================================================
    ax = fig.add_subplot(gs[2, 0])
    show_img(ax, paths["r3_input"],
             title="Input (masked)",
             ylabel="(c) Tolerance Trade-off\n(COCO, with captions)")

    ax = fig.add_subplot(gs[2, 1])
    show_img(ax, paths["r3_baseline_def"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")
    add_setting_label(ax, "Setting A: Default τ=(0.03, 0.3)",
                      color="darkblue")

    ax = fig.add_subplot(gs[2, 2])
    show_img(ax, paths["r3_freqspec_def"], title="FreqSpec",
             title_color="darkblue")
    add_metric_box(ax, "PSNR 20.3\nLPIPS 0.07\n1.19×")

    ax = fig.add_subplot(gs[2, 3])
    show_img(ax, paths["r3_baseline_strict"], title="Baseline (target only)")
    add_metric_box(ax, "Reference")
    add_setting_label(ax, "Setting B: Strict τ=(0.01, 0.1)",
                      color="darkblue")

    ax = fig.add_subplot(gs[2, 4])
    show_img(ax, paths["r3_freqspec_strict"], title="FreqSpec",
             title_color="darkblue")
    add_metric_box(ax, "PSNR 24.9\nLPIPS 0.04\n1.00×")

    fig.suptitle(
        "Figure 5: Qualitative Results — Each FreqSpec output paired with its own baseline",
        fontsize=14, fontweight="bold", y=0.99
    )

    out_png = "/mnt/HDD_12TB/bam_ki/results/qualitative_figure5.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"saved: {out_png}")
    print(f"saved: {out_png.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()