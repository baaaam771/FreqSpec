#!/usr/bin/env python
"""
make_lookahead_diagram.py — Illustrative K-step gating diagram.

No inference; pure matplotlib. Produces a single PDF/PNG figure that
explains:
  - Phase 1 (t/T > 0.7): target-only warm-up steps.
  - Phase 2 (t/T <= 0.7): verified timesteps (target + draft), with
    K-1 draft-only lookahead steps allowed only when block-head drift
    is small and acceptance rate is high.
  - Otherwise: spec-1 (single-step) advance and re-verify.

Usage:
    python make_lookahead_diagram.py --out_path figures/lookahead_diagram
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_path", default="figures/lookahead_diagram")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    # Schedule layout: 11 timesteps in our schematic
    # t_norm from 1.0 -> 0.0 (left to right)
    # marker style per step:
    #  - target-only warmup (Phase 1)
    #  - verified (target + draft, every block head)
    #  - draft-only lookahead (K-1 steps after a successful gate)
    #  - spec-1 (one-step advance when gate fails)

    K = args.K
    steps = list(range(15))  # 15 ticks
    # assign phase labels
    labels = []
    for i, _ in enumerate(steps):
        if i < 4:
            labels.append("warmup")  # Phase 1, target-only
        else:
            labels.append(None)  # filled below

    # Phase 2 plan:
    # i=4  verified (block head)
    # i=5  draft-only  (lookahead inside block)
    # i=6  draft-only  (lookahead, K-1 = 2 inner steps)
    # i=7  verified (next block head) — gate fails -> spec-1
    # i=8  verified (gate fails again)
    # i=9  verified (gate succeeds) — block head
    # i=10 draft-only
    # i=11 draft-only
    # i=12 verified (final block head; gate fails)
    # i=13 verified (last verified step)
    # i=14 verified (last)
    pattern = [
        ("warmup", ""),
        ("warmup", ""), ("warmup", ""), ("warmup", ""),
        ("verified",  "block head"),
        ("draft",     "draft-only"),
        ("draft",     "draft-only"),
        ("verified",  "block head\n(gate fails)"),
        ("verified",  "spec-1"),
        ("verified",  "block head"),
        ("draft",     "draft-only"),
        ("draft",     "draft-only"),
        ("verified",  "block head\n(gate fails)"),
        ("verified",  "spec-1"),
        ("verified",  "final"),
    ]
    n = len(pattern)

    fig, ax = plt.subplots(figsize=(13, 4.6))

    # Color scheme
    color = {
        "warmup":   "#bdbdbd",   # gray
        "verified": "#1f77b4",   # blue
        "draft":    "#2ca02c",   # green
    }
    edge = {
        "warmup":   "#737373",
        "verified": "#0b3d70",
        "draft":    "#0f5e1d",
    }

    # Y positions
    y_axis = 0.65
    box_h = 0.32
    box_w = 0.82
    txt_y_above = y_axis + box_h / 2 + 0.10
    txt_y_below = y_axis - box_h / 2 - 0.55  # bracket goes here, below tick labels

    # Draw cells
    for i, (kind, note) in enumerate(pattern):
        x = i
        rect = Rectangle(
            (x - box_w / 2, y_axis - box_h / 2), box_w, box_h,
            facecolor=color[kind], edgecolor=edge[kind], linewidth=1.2,
        )
        ax.add_patch(rect)
        # tick label (step index)
        ax.text(x, y_axis - box_h / 2 - 0.07, f"t$_{{{i}}}$",
                ha="center", va="top", fontsize=9)
        # phase note above
        if note:
            ax.text(x, txt_y_above, note,
                    ha="center", va="bottom",
                    fontsize=8.4, color=edge[kind])

    # Phase 1 / Phase 2 separator
    sep_x = 3.5
    ax.axvline(sep_x, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(1.5, y_axis + box_h / 2 + 0.55,
            "Phase 1: target-only ($t/T > 0.7$)",
            ha="center", fontsize=10)
    ax.text(9.0, y_axis + box_h / 2 + 0.55,
            "Phase 2: verified timesteps + $K{-}1$ draft-only lookahead",
            ha="center", fontsize=10)

    # K-block annotations using simple line+text (no curved arrows that overlap)
    def bracket(x_start, x_end, label, y_top):
        # horizontal line
        ax.plot([x_start - 0.42, x_end + 0.42], [y_top, y_top],
                color="#444", linewidth=1.2)
        # short verticals at endpoints
        ax.plot([x_start - 0.42, x_start - 0.42],
                [y_top, y_top + 0.05], color="#444", linewidth=1.2)
        ax.plot([x_end + 0.42, x_end + 0.42],
                [y_top, y_top + 0.05], color="#444", linewidth=1.2)
        ax.text((x_start + x_end) / 2, y_top - 0.13,
                label, ha="center", va="top",
                fontsize=9, color="#222")

    # Bracket: K=3 block at i=4..6 and i=9..11
    bracket(4, 6, f"$K{{=}}{K}$ block (1 verify + {K-1} draft-only)", txt_y_below)
    bracket(9, 11, f"$K{{=}}{K}$ block", txt_y_below)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color["warmup"], label="Warmup (target only)"),
        mpatches.Patch(color=color["verified"], label="Verified step (target+draft)"),
        mpatches.Patch(color=color["draft"], label="Draft-only lookahead"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.18))

    # Axis cleanup
    ax.set_xlim(-1.0, n)
    ax.set_ylim(txt_y_below - 0.45, txt_y_above + 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Direction arrow: t decreasing left to right
    arrow_y = txt_y_below - 0.30
    ax.annotate(
        "", xy=(n - 0.5, arrow_y), xycoords="data",
        xytext=(-0.5, arrow_y), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#666", linewidth=1.0),
    )
    ax.text(n / 2, arrow_y - 0.10,
            r"reverse time ($t$ decreasing)",
            ha="center", va="top", fontsize=9, color="#666")

    plt.tight_layout()
    out = args.out_path.rstrip(".pdf").rstrip(".png")
    plt.savefig(out + ".pdf", bbox_inches="tight", dpi=args.dpi)
    plt.savefig(out + ".png", bbox_inches="tight", dpi=args.dpi)
    print(f"[done] saved {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
