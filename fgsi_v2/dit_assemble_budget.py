#!/usr/bin/env python
"""
dit_assemble_budget.py — collect the DiT accept-budget sweep into a table.

Reads sweep_r{0.3,0.5,0.7,0.9}/sampling_summary.json and emits a LaTeX table and
CSV showing, at each budget: FreqSpec-token vs Random-token (accept, target-token
usage, pixel-std), with the constant Target-only / Draft-only references. The
message for reviewers: the FreqSpec > Random ordering and the controllable
target-usage / pixel-std trade-off hold across budgets, not at one lucky threshold.

Usage:
    python dit_assemble_budget.py \
        --root /mnt/HDD_12TB/bam_ki/results/dit_token_poc_nano \
        --ratios 0.3 0.5 0.7 0.9 \
        --out_dir /mnt/HDD_12TB/bam_ki/results/dit_token_poc_nano/budget
"""
import argparse
import csv
import json
import os


def _load(root, r):
    p = os.path.join(root, f"sweep_r{r}", "sampling_summary.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p)).get("methods", {})


def main(args):
    rows = []           # (method, accept, tgt_use, px_std)
    ref_target = ref_draft = None
    fs_by_r, rnd_by_r = {}, {}
    for r in args.ratios:
        m = _load(args.root, r)
        if not m:
            print(f"[budget] missing sweep_r{r}; skipping")
            continue
        if ref_target is None and "target" in m:
            t = m["target"]
            ref_target = (t.get("target_token_usage", 1.0), t.get("pixel_std"))
        if ref_draft is None and "draft" in m:
            d = m["draft"]
            ref_draft = (d.get("target_token_usage", 0.0), d.get("pixel_std"))
        if "freqspec" in m:
            f = m["freqspec"]
            fs_by_r[r] = (f.get("accept"), f.get("target_token_usage"), f.get("pixel_std"))
        if "random" in m:
            g = m["random"]
            rnd_by_r[r] = (g.get("accept"), g.get("target_token_usage"), g.get("pixel_std"))

    os.makedirs(args.out_dir, exist_ok=True)
    # CSV
    csv_path = os.path.join(args.out_dir, "budget.csv")
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["method", "accept", "target_tokens", "pixel_std"])
        if ref_target:
            w.writerow(["Target-only", "", f"{ref_target[0]*100:.0f}%", ref_target[1]])
        if ref_draft:
            w.writerow(["Draft-only", "", f"{ref_draft[0]*100:.0f}%", ref_draft[1]])
        for r in args.ratios:
            if r in fs_by_r:
                a, tu, ps = fs_by_r[r]
                w.writerow([f"FreqSpec (acc {r})", f"{a:.3f}", f"{tu*100:.0f}%", ps])
        for r in args.ratios:
            if r in rnd_by_r:
                a, tu, ps = rnd_by_r[r]
                w.writerow([f"Random (acc {r})", f"{a:.3f}", f"{tu*100:.0f}%", ps])
    print(f"[budget] wrote {csv_path}")

    # LaTeX
    tex_path = os.path.join(args.out_dir, "budget_table.tex")
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(r"\begin{tabular}{lrrr}")
    L.append(r"\toprule")
    L.append(r"Method & Accept & Target tok. & pixel-std \\")
    L.append(r"\midrule")
    if ref_target:
        L.append(rf"Target-only (DiT-S) & -- & {ref_target[0]*100:.0f}\% & {ref_target[1]:.3f} \\")
    if ref_draft:
        L.append(rf"Draft-only (DiT-Nano) & -- & {ref_draft[0]*100:.0f}\% & {ref_draft[1]:.3f} (coll.) \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{l}{\itshape FreqSpec-token (agreement)} \\")
    for r in args.ratios:
        if r in fs_by_r:
            a, tu, ps = fs_by_r[r]
            L.append(rf"\quad accept {r} & {a:.3f} & {tu*100:.0f}\% & {ps:.3f} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{l}{\itshape Random-token (same budget)} \\")
    for r in args.ratios:
        if r in rnd_by_r:
            a, tu, ps = rnd_by_r[r]
            L.append(rf"\quad accept {r} & {a:.3f} & {tu*100:.0f}\% & {ps:.3f} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\caption{\textbf{DiT token-mixing accept-budget sweep} (CIFAR-10, "
             r"DiT-Nano draft). As the accept ratio rises, target-token usage falls "
             r"and pixel-std moves from target-like toward draft-like; FreqSpec-token "
             r"stays closer to the target than Random-token at every budget, so the "
             r"agreement advantage is not tied to one threshold.}")
    L.append(r"\label{tab:dit_budget}")
    L.append(r"\end{table}")
    open(tex_path, "w").write("\n".join(L) + "\n")
    print(f"[budget] wrote {tex_path}")
    # console echo
    print("[budget] FreqSpec px-std by accept:",
          "  ".join(f"{r}:{fs_by_r[r][2]:.3f}" for r in args.ratios if r in fs_by_r))
    print("[budget] Random   px-std by accept:",
          "  ".join(f"{r}:{rnd_by_r[r][2]:.3f}" for r in args.ratios if r in rnd_by_r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--ratios", type=str, nargs="+", default=["0.3", "0.5", "0.7", "0.9"])
    ap.add_argument("--out_dir", type=str, required=True)
    main(ap.parse_args())
