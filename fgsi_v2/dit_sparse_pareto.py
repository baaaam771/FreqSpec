#!/usr/bin/env python
"""
dit_sparse_pareto.py — paper figures from sparse_results.csv.

Figure 1 (money plot): TOTAL compute (relative to 50-step dense) vs FID.
    - dense step-reduction curve (the honest reference)
    - cache_attn families (oracle / anchor / random) at s50 and short schedules
    - legacy freeze modes (sparse_mlp / sparse_attn) in grey, showing the
      execution-mechanism gap
    - mix ceilings as horizontal dashed levels per budget

Figure 2 (ceiling closure): per budget r, bars for [mix ceiling | cache_attn
oracle | cache_attn anchor | freeze sparse_attn] — the "execution solved"
evidence: cache reaches the mixing ceiling at a fraction of its compute.

Usage:
    python dit_sparse_pareto.py --root .../sweep --out .../figs
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(root):
    rows = []
    with open(os.path.join(root, "sparse_results.csv")) as f:
        for r in csv.DictReader(f):
            if not r.get("fid"):
                continue
            for k in ("split", "hard_ratio", "total_rel", "fid", "dense_until"):
                r[k] = float(r[k])
            for k in ("steps", "refresh_every"):
                r[k] = int(float(r[k]))
            rows.append(r)
    return rows


def main(args):
    rows = load(args.root)
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------- Figure 1
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    dense = sorted([(r["total_rel"], r["fid"]) for r in rows
                    if r["selector"] == "dense"])
    ax.plot(*zip(*dense), "o-", c="crimson", lw=2, label="dense (reduced steps)",
            zorder=5)
    fam = defaultdict(list)
    for r in rows:
        if r["suffix_mode"] == "cache_attn":
            fam[(r["selector"], "s50" if r["steps"] == 50 else "short")].append(
                (r["total_rel"], r["fid"]))
        elif r["suffix_mode"] in ("sparse_mlp", "sparse_attn") \
                and r["selector"] == "oracle" and r["refresh_every"] == 0 \
                and r["dense_until"] == 1.0:
            fam[("freeze", "")].append((r["total_rel"], r["fid"]))
    style = {("oracle", "s50"): ("tab:blue", "o", "cache oracle (50 steps)"),
             ("oracle", "short"): ("tab:blue", "^", "cache oracle (25-30 steps)"),
             ("anchor", "s50"): ("tab:green", "s", "cache anchor (50 steps)"),
             ("anchor", "short"): ("tab:green", "v", "cache anchor (25-30 steps)"),
             ("random", "s50"): ("tab:orange", "x", "cache random"),
             ("freeze", ""): ("grey", ".", "frozen context (A1/A2)")}
    for key, pts in fam.items():
        if key not in style:
            continue
        c, mk, lb = style[key]
        pts = sorted(pts)
        ax.scatter(*zip(*pts), c=c, marker=mk, s=45, label=lb, alpha=0.85)
    mixes = [r for r in rows if r["selector"] == "mix"]
    for r in mixes:
        ax.axhline(r["fid"], ls="--", c="purple", lw=1, alpha=0.6)
        ax.annotate(f"mix ceiling r={r['hard_ratio']:g} (s{r['steps']})",
                    (0.02, r["fid"]), fontsize=7, color="purple",
                    va="bottom")
    ax.set_xlabel("total compute vs 50-step dense target (MACs)")
    ax.set_ylabel("FID-10k")
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Sparse verifier execution vs step reduction (ImageNet-64)")
    fig.tight_layout()
    p1 = os.path.join(args.out, "pareto_total_vs_fid.png")
    fig.savefig(p1, dpi=200)
    print(f"[pareto] wrote {p1}")

    # ------------------------------------------------------------- Figure 2
    budgets = sorted({r["hard_ratio"] for r in rows if r["selector"] == "mix"})
    if budgets:
        fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
        kinds = [("mix ceiling", "purple"), ("cache oracle", "tab:blue"),
                 ("cache anchor", "tab:green"), ("frozen attn", "grey")]
        width = 0.2
        for bi, rr in enumerate(budgets):
            def pick(sel, mode):
                c = [r["fid"] for r in rows if r["selector"] == sel
                     and r["suffix_mode"] == mode and r["hard_ratio"] == rr
                     and r["steps"] == 50 and r["refresh_every"] == 0
                     and r["dense_until"] == 1.0]
                return min(c) if c else None
            vals = [pick("mix", "sparse_mlp") or pick("mix", "cache_attn"),
                    pick("oracle", "cache_attn"),
                    pick("anchor", "cache_attn"),
                    pick("oracle", "sparse_attn")]
            for ki, (v, (lb, c)) in enumerate(zip(vals, kinds)):
                if v is None:
                    continue
                ax2.bar(bi + (ki - 1.5) * width, v, width, color=c,
                        label=lb if bi == 0 else None)
        ax2.set_xticks(range(len(budgets)))
        ax2.set_xticklabels([f"r={b:g}" for b in budgets])
        ax2.set_ylabel("FID-10k")
        ax2.legend(fontsize=8)
        ax2.set_title("Ceiling closure: cached execution reaches the mixing "
                      "ceiling")
        fig2.tight_layout()
        p2 = os.path.join(args.out, "ceiling_closure.png")
        fig2.savefig(p2, dpi=200)
        print(f"[pareto] wrote {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    main(ap.parse_args())
