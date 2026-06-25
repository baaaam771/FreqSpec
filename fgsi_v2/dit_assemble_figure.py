#!/usr/bin/env python
"""
dit_assemble_figure.py — Phase 3 PoC figure: 4-way DiT sample-grid comparison.

Lays the target / draft / freqspec / random sample grids side by side with
per-method annotations (accept, target-token usage, pixel std) read from
sampling_summary.json. The message: the draft alone collapses, the token-wise
agreement verifier (freqspec) restores target-level samples while using the
target on only ~30% of tokens, and random mixing at the same ratio is clearly
worse — so the agreement-based selection, not merely mixing, is what matters.

Usage:
    python dit_assemble_figure.py \
        --grid_dir results/dit_token_poc_nano/sampling \
        --out results/dit_token_poc_nano/dit_poc_grids.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ORDER = ["target", "draft", "freqspec", "random"]
TITLES = {"target": "Target-only (DiT-S)", "draft": "Draft-only (DiT-Nano)",
          "freqspec": "FreqSpec-token", "random": "Random-token"}


def main(args):
    summ = {}
    sp = os.path.join(args.grid_dir, "sampling_summary.json")
    if os.path.isfile(sp):
        summ = json.load(open(sp)).get("methods", {})

    imgs = {}
    for m in ORDER:
        p = os.path.join(args.grid_dir, f"grid_{m}.png")
        if os.path.isfile(p):
            imgs[m] = mpimg.imread(p)
    present = [m for m in ORDER if m in imgs]
    if not present:
        print(f"[dit-fig] no grids found in {args.grid_dir}"); return

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 3.5))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, present):
        ax.imshow(imgs[m]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(TITLES[m], fontsize=10)
        s = summ.get(m, {})
        if m == "freqspec":
            cap = (f"accept {s.get('accept', 0):.2f}\n"
                   f"target tokens {s.get('target_token_usage', 0)*100:.0f}%")
        elif m == "random":
            cap = (f"accept {s.get('accept', 0):.2f}\n(same ratio, random)")
        elif m == "draft":
            cap = "collapsed"
        else:
            cap = "reference"
        if s:
            cap += f"\npx-std {s.get('pixel_std', 0):.3f}"
        ax.set_xlabel(cap, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[dit-fig] saved {args.out} and {pdf}")
    if summ:
        print("[dit-fig] order target>=freqspec>random>>draft by px-std:",
              "  ".join(f"{m}={summ.get(m, {}).get('pixel_std', 0):.3f}" for m in present))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="dit_poc_grids.png")
    ap.add_argument("--dpi", type=int, default=160)
    main(ap.parse_args())
