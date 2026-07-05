#!/usr/bin/env python
"""
dit_inpaint_assemble.py — collect metrics.json from a sweep of
dit_inpaint_sampler.py output dirs into one comparison table (markdown +
csv + LaTeX booktabs), sorted by MACs.

Usage:
    python dit_inpaint_assemble.py \
        --root /mnt/HDD_12TB/bam_ki/results/dit_inp \
        --out  /mnt/HDD_12TB/bam_ki/results/dit_inp/table
"""
import argparse
import glob
import json
import os


COLS = [("method", "{:s}"), ("macs_vs_dense50", "{:.3f}"),
        ("tgt_dense_nfe", "{:.1f}"), ("tgt_sparse_nfe", "{:.1f}"),
        ("drf_nfe", "{:.1f}"), ("hard_k", "{:.0f}"),
        ("mask_psnr", "{:.3f}"), ("mask_lpips", "{:.4f}"),
        ("mask_mse_t", "{:.5f}"), ("mask_lpips_t", "{:.4f}")]


def label(cfg):
    if cfg["mode"] == "dense":
        return f"dense s{cfg['steps']}"
    if cfg["mode"] == "draft":
        return "draft only"
    if cfg["mode"] == "mix":
        return f"mix ceiling r={cfg['hard_ratio']}"
    r = cfg["hard_ratio"] if cfg["hard_ratio"] > 0 else "auto"
    tag = f"dace {cfg['selector']} r={r} c={cfg['cache_period']} " \
          f"m={cfg['split_m']} easy={cfg['easy']} S={cfg['steps']}"
    if cfg.get("block", 1) > 1:
        tag += f" blk{cfg['block']}"
    if cfg.get("restrict_to_mask"):
        tag += " inmask"
    return tag


def main(args):
    rows = []
    for mj in sorted(glob.glob(os.path.join(args.root, "**", "metrics.json"),
                               recursive=True)):
        with open(mj) as f:
            r = json.load(f)
        r["method"] = label(r["config"])
        rows.append(r)
    rows.sort(key=lambda r: r.get("macs_vs_dense50", 9e9))
    os.makedirs(args.out, exist_ok=True)

    def _fmt(r, k, fm):
        v = r.get(k)
        return "-" if v is None else fm.format(v)

    # markdown
    md = ["| " + " | ".join(k for k, _ in COLS) + " |",
          "|" + "---|" * len(COLS)]
    for r in rows:
        md.append("| " + " | ".join(_fmt(r, k, fm) for k, fm in COLS) + " |")
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    # csv
    with open(os.path.join(args.out, "table.csv"), "w") as f:
        f.write(",".join(k for k, _ in COLS) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r, k, fm) for k, fm in COLS) + "\n")
    # latex
    tex = ["\\begin{tabular}{l" + "r" * (len(COLS) - 1) + "}", "\\toprule",
           " & ".join(k.replace("_", "\\_") for k, _ in COLS) + " \\\\",
           "\\midrule"]
    for r in rows:
        tex.append(" & ".join(_fmt(r, k, fm) for k, fm in COLS) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(args.out, "table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\n[done] wrote table.md / table.csv / table.tex to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    main(ap.parse_args())
