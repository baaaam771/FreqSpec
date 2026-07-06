#!/usr/bin/env python
"""
dit_inpaint_assemble.py — collect metrics.json from a sweep into comparison
tables (markdown + csv + LaTeX). REVISED for the new schema:
  - metrics live under "overall"; compute under "compute"
  - metric names follow the review (*_to_dense50, known_*)
  - block selection labelled "block-structured selection" (item 6)
  - optional per-bucket table (item 12) with --by_bucket

Usage:
    python dit_inpaint_assemble.py --root results/dit_inp --out results/dit_inp/table
    python dit_inpaint_assemble.py --root results/dit_inp --out results/dit_inp/table --by_bucket
"""
import argparse
import glob
import json
import os


COLS = [("method", "{:s}"), ("executed_macs_vs_dense50", "{:.3f}"),
        ("ideal_macs_vs_dense50", "{:.3f}"),
        ("tgt_dense_nfe", "{:.1f}"), ("tgt_sparse_nfe", "{:.1f}"),
        ("drf_nfe", "{:.1f}"), ("mean_true_k", "{:.0f}"),
        ("mean_executed_k", "{:.0f}"),
        ("mask_psnr", "{:.3f}"), ("known_psnr", "{:.2f}"),
        ("known_ssim", "{:.4f}"), ("mask_lpips", "{:.4f}"),
        ("mask_mse_to_dense50", "{:.5f}"),
        ("mask_lpips_to_dense50", "{:.4f}"),
        ("boundary_lpips_to_dense50", "{:.4f}")]


def label(cfg):
    if cfg["mode"] == "dense":
        return f"dense s{cfg['steps']}"
    if cfg["mode"] == "draft":
        return "draft only"
    if cfg["mode"] == "mix":
        return f"mix ceiling r={cfg['hard_ratio']}"
    reg = cfg.get("region", "mask")
    bud = cfg.get("budget", "ratio")
    r = "exact" if bud == "mask_exact" else f"r={cfg['hard_ratio']}"
    tag = (f"dace[{cfg.get('suffix','cache')}] {reg}/{cfg['selector']} {r} "
           f"freq={cfg.get('freq_src','-')} c={cfg['cache_period']} "
           f"m={cfg['split_m']} easy={cfg['easy']} S={cfg['steps']}")
    if cfg.get("block", 1) > 1:
        tag += f" +{cfg['block']}x{cfg['block']}block-struct-sel"
    if cfg.get("no_reinject"):
        tag += " noReinj"
    return tag


def flat(r):
    """Flatten one metrics.json into a row dict."""
    row = {"method": label(r["config"])}
    row.update(r.get("overall", {}))
    row.update(r.get("compute", {}))
    return row


def _fmt(row, k, fm):
    v = row.get(k)
    return "-" if v is None else fm.format(v)


def write_tables(rows, out, name):
    os.makedirs(out, exist_ok=True)
    rows.sort(key=lambda r: r.get("executed_macs_vs_dense50", 9e9) or 9e9)
    md = ["| " + " | ".join(k for k, _ in COLS) + " |",
          "|" + "---|" * len(COLS)]
    for r in rows:
        md.append("| " + " | ".join(_fmt(r, k, fm) for k, fm in COLS) + " |")
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out, f"{name}.csv"), "w") as f:
        f.write(",".join(k for k, _ in COLS) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r, k, fm) for k, fm in COLS) + "\n")
    tex = ["\\begin{tabular}{l" + "r" * (len(COLS) - 1) + "}", "\\toprule",
           " & ".join(k.replace("_", "\\_") for k, _ in COLS) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(_fmt(r, k, fm) for k, fm in COLS) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))


def main(args):
    metas = []
    for mj in sorted(glob.glob(os.path.join(args.root, "**", "metrics.json"),
                               recursive=True)):
        with open(mj) as f:
            metas.append(json.load(f))
    write_tables([flat(m) for m in metas], args.out, "table")
    if args.by_bucket:
        for b in ("small", "medium", "large"):
            rows = []
            for m in metas:
                row = {"method": label(m["config"])}
                row.update(m.get("by_bucket", {}).get(b, {}))
                row.update(m.get("compute", {}))
                if row.get("n"):
                    rows.append(row)
            print(f"\n=== bucket: {b} ===")
            write_tables(rows, args.out, f"table_{b}")
    print(f"\n[done] tables in {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--by_bucket", action="store_true")
    main(ap.parse_args())
