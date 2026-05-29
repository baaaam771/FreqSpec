#!/usr/bin/env python
"""
analyze_speed_matched.py — Compute region-aware metrics and build the
speed-matched comparison table from baseline_sweep.py outputs.

Workflow:
  1. Read each method's results.csv (timing, NFE) from the sweep output.
  2. For every (method, image), load gt.png / out.png / mask.png and compute
     PSNR, SSIM, LPIPS, masked LPIPS, boundary LPIPS via metrics_extended.
  3. Aggregate per-method means + std.
  4. Compute speedup = target_s50_time / method_time (per image, then mean).
  5. Find speed-matched pairs: for each FreqSpec preset, find the reduced-step
     target with the closest mean speedup, and print them side by side.
  6. Emit a LaTeX table (BMVC style) ready to paste.

Example:
    python analyze_speed_matched.py \\
        --sweep_root /mnt/HDD_12TB/bam_ki/results/baseline_sweep_places2 \\
        --dataset_name Places2 \\
        --boundary_k 8
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from metrics_extended import RegionMetrics


def load_png(path):
    return np.array(Image.open(path).convert("RGB"))


def load_mask_png(path):
    m = np.array(Image.open(path).convert("L"))
    return (m > 127).astype(np.float32)


def read_method_csv(method_dir):
    """Return list of row dicts from a method's results.csv."""
    csv_path = os.path.join(method_dir, "results.csv")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main(args):
    sweep_root = Path(args.sweep_root)
    method_dirs = sorted(
        d for d in sweep_root.iterdir()
        if d.is_dir() and os.path.isfile(d / "results.csv")
    )
    print(f"[analyze] found {len(method_dirs)} methods: "
          f"{[d.name for d in method_dirs]}")

    metric_engine = RegionMetrics(device=args.device, boundary_k=args.boundary_k)

    # Pre-load the full-step target outputs (target_s50) as the reference for
    # target-deviation metrics. Key by image idx.
    ref_name = f"target_s{args.ref_steps}"
    ref_dir = sweep_root / ref_name
    target_ref = {}
    if ref_dir.is_dir():
        for r in read_method_csv(ref_dir):
            idx = int(r["idx"])
            op = ref_dir / f"img_{idx:03d}" / "out.png"
            if op.exists():
                target_ref[idx] = load_png(op)
        print(f"[analyze] loaded {len(target_ref)} target references "
              f"from {ref_name} (for target-deviation metrics)")
    else:
        print(f"[analyze] WARNING: {ref_name} not found; "
              f"target-deviation metrics will be skipped")

    # per-method: lists of metrics + timing
    agg = defaultdict(lambda: defaultdict(list))

    for mdir in method_dirs:
        method = mdir.name
        rows = read_method_csv(mdir)
        for r in rows:
            idx = int(r["idx"])
            img_dir = mdir / f"img_{idx:03d}"
            gt_p = img_dir / "gt.png"
            out_p = img_dir / "out.png"
            mask_p = img_dir / "mask.png"
            if not (gt_p.exists() and out_p.exists() and mask_p.exists()):
                continue
            gt = load_png(gt_p)
            out = load_png(out_p)
            mask = load_mask_png(mask_p)

            # (A) ground-truth-based metrics (restoration quality)
            m = metric_engine.compute(out, gt, mask)
            for k, v in m.items():
                agg[method][k].append(v)

            # (B) target-deviation metrics (target fidelity, original criterion)
            if idx in target_ref:
                mt = metric_engine.compute_vs_target(out, target_ref[idx], mask)
                for k, v in mt.items():
                    agg[method][k].append(v)

            agg[method]["time_sec"].append(float(r["time_sec"]))
            agg[method]["target_nfe"].append(float(r["target_nfe"]))
            if r["draft_nfe"]:
                agg[method]["draft_nfe"].append(float(r["draft_nfe"]))
            if r["accept_rate"]:
                agg[method]["accept_rate"].append(float(r["accept_rate"]))

    # reference time = target_s50 mean time (for speedup)
    ref_key = None
    for cand in ["target_s50", "target_s50".replace("50", str(args.ref_steps))]:
        if cand in agg:
            ref_key = cand
            break
    if ref_key is None:
        # fall back: slowest method = reference
        ref_key = max(agg, key=lambda k: np.mean(agg[k]["time_sec"]))
    ref_time = np.mean(agg[ref_key]["time_sec"])
    print(f"[analyze] reference = {ref_key}, mean time = {ref_time:.3f}s")

    # build summary
    summary = {}
    for method, d in agg.items():
        n = len(d["lpips"])
        speedup = ref_time / np.mean(d["time_sec"])
        entry = {
            "n": n,
            "speedup": speedup,
            "time": np.mean(d["time_sec"]),
            "psnr": np.mean(d["psnr"]),
            "ssim": np.mean(d["ssim"]),
            "lpips": np.mean(d["lpips"]),
            "lpips_std": np.std(d["lpips"]),
            "masked_lpips": np.mean(d["masked_lpips"]),
            "boundary_lpips": np.mean(d["boundary_lpips"]),
            "masked_psnr": np.mean(d["masked_psnr"]),
            "accept_rate": np.mean(d["accept_rate"]) if d["accept_rate"] else None,
        }
        # target-deviation metrics (if target reference was available)
        if d["lpips_vs_tgt"]:
            entry["lpips_vs_tgt"] = np.mean(d["lpips_vs_tgt"])
            entry["psnr_vs_tgt"] = np.mean(d["psnr_vs_tgt"])
            entry["masked_lpips_vs_tgt"] = np.mean(d["masked_lpips_vs_tgt"])
            entry["boundary_lpips_vs_tgt"] = np.mean(d["boundary_lpips_vs_tgt"])
        summary[method] = entry

    has_vs_tgt = any("lpips_vs_tgt" in s for s in summary.values())

    # ---- print full summary: (A) ground-truth-based ----
    print(f"\n{'='*100}")
    print(f"(A) GROUND-TRUTH METRICS — {args.dataset_name}  "
          f"[restoration quality vs original image]")
    print(f"{'='*100}")
    hdr = (f"{'method':22} {'n':>4} {'speedup':>8} {'time':>7} "
           f"{'PSNR':>6} {'SSIM':>6} {'LPIPS':>7} {'mLPIPS':>7} "
           f"{'bLPIPS':>7} {'accept':>7}")
    print(hdr)
    print("-" * len(hdr))
    for method in sorted(summary, key=lambda k: -summary[k]["speedup"]):
        s = summary[method]
        acc = f"{s['accept_rate']:.3f}" if s["accept_rate"] is not None else "  -  "
        print(f"{method:22} {s['n']:>4} {s['speedup']:>7.2f}x {s['time']:>6.2f}s "
              f"{s['psnr']:>6.2f} {s['ssim']:>6.3f} {s['lpips']:>7.4f} "
              f"{s['masked_lpips']:>7.4f} {s['boundary_lpips']:>7.4f} {acc:>7}")

    # ---- print full summary: (B) target-deviation ----
    if has_vs_tgt:
        print(f"\n{'='*100}")
        print(f"(B) TARGET-DIVERGENCE METRICS — {args.dataset_name}  "
              f"[divergence from {ref_name} output]")
        print(f"    Lower = closer to target (fidelity view).")
        print(f"    Higher = more diverse plausible completion (diversity view).")
        print(f"    For inpainting (multi-modal task), high divergence is NOT")
        print(f"    inherently bad — it can indicate semantic diversity.")
        print(f"{'='*100}")
        hdr2 = (f"{'method':22} {'speedup':>8} "
                f"{'LPIPS_t':>8} {'PSNR_t':>7} {'mLPIPS_t':>9} {'bLPIPS_t':>9}")
        print(hdr2)
        print("-" * len(hdr2))
        for method in sorted(summary, key=lambda k: -summary[k]["speedup"]):
            s = summary[method]
            if "lpips_vs_tgt" not in s:
                continue
            print(f"{method:22} {s['speedup']:>7.2f}x "
                  f"{s['lpips_vs_tgt']:>8.4f} {s['psnr_vs_tgt']:>7.2f} "
                  f"{s['masked_lpips_vs_tgt']:>9.4f} "
                  f"{s['boundary_lpips_vs_tgt']:>9.4f}")
        print(f"\n  Note: target_s{args.ref_steps} vs itself = 0 (it is the "
              f"reference).")
        print(f"  For target-fidelity scenarios (e.g. video frame consistency):")
        print(f"    lower divergence = better.")
        print(f"  For inpainting quality (multi-modal, no single right answer):")
        print(f"    divergence alone does not measure quality — combine with")
        print(f"    no-reference IQA (CLIP-IQA, MUSIQ) and FID to assess.")

    # ---- speed-matched pairs ----
    print(f"\n{'='*100}")
    print("SPEED-MATCHED COMPARISON (FreqSpec vs closest reduced-step target)")
    print(f"{'='*100}")
    fs_methods = [m for m in summary if m.startswith("freqspec")]
    tgt_methods = [m for m in summary if m.startswith("target")]

    pairs = []
    for fs in fs_methods:
        fs_sp = summary[fs]["speedup"]
        # closest target by speedup
        closest = min(tgt_methods,
                      key=lambda t: abs(summary[t]["speedup"] - fs_sp))
        pairs.append((fs, closest))
        sf, st = summary[fs], summary[closest]
        print(f"\n  {fs} (speedup {sf['speedup']:.2f}x) "
              f"vs {closest} (speedup {st['speedup']:.2f}x):")
        print(f"    {'metric':22} {'FreqSpec':>10} {'Target':>10} {'winner':>8}")
        # (A) ground-truth metrics
        gt_metrics = [("lpips", True), ("masked_lpips", True),
                      ("boundary_lpips", True), ("psnr", False), ("ssim", False)]
        print(f"    -- ground-truth (restoration quality) --")
        for k, lower_better in gt_metrics:
            fv, tv = sf[k], st[k]
            win = ("FreqSpec" if (fv < tv) == lower_better else "Target")
            print(f"    {k:22} {fv:>10.4f} {tv:>10.4f} {win:>8}")
        # (B) target-divergence metrics — interpret BOTH ways
        if "lpips_vs_tgt" in sf and "lpips_vs_tgt" in st:
            print(f"    -- target-divergence (vs {ref_name}, two views) --")
            for k in ["lpips_vs_tgt", "masked_lpips_vs_tgt",
                      "boundary_lpips_vs_tgt"]:
                fv, tv = sf[k], st[k]
                # both interpretations
                fid_winner = "FreqSpec" if fv < tv else "Target"
                div_winner = "FreqSpec" if fv > tv else "Target"
                annot = f"fid:{fid_winner} / div:{div_winner}"
                print(f"    {k:22} {fv:>10.4f} {tv:>10.4f}  {annot}")

    # ---- LaTeX table ----
    latex = build_latex_table(summary, pairs, args.dataset_name)
    out_tex = os.path.join(args.sweep_root, "speed_matched_table.tex")
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"\n[analyze] LaTeX table -> {out_tex}")

    # ---- save summary json ----
    out_json = os.path.join(args.sweep_root, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[analyze] summary -> {out_json}")


def build_latex_table(summary, pairs, dataset_name):
    has_vs_tgt = any("lpips_vs_tgt" in summary[m] for m, _ in pairs)

    def one_table(metric_set, caption, label_suffix, header_cols):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{" + caption + "}",
            r"\label{tab:" + label_suffix + "_" + dataset_name.lower() + "}",
            r"\begin{tabular}{ll" + "c" * len(header_cols) + "}",
            r"\toprule",
            r"Method & Steps & " + " & ".join(header_cols) + r" \\",
            r"\midrule",
        ]
        seen = set()
        for fs, tgt in pairs:
            for method in [tgt, fs]:
                if method in seen:
                    continue
                seen.add(method)
                s = summary[method]
                steps = "50" if method.startswith("freqspec") else \
                    method.replace("target_s", "")
                name = method.replace("_", r"\_")
                bold = method.startswith("freqspec")

                def fmt(x, d=4):
                    txt = f"{x:.{d}f}"
                    return (r"\textbf{" + txt + "}") if bold else txt

                cells = [f"{name}", f"{steps}",
                         f"{s['speedup']:.2f}$\\times$"]
                for key, d in metric_set:
                    cells.append(fmt(s[key], d) if key in s else "--")
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    # (A) ground-truth table
    gt_tab = one_table(
        metric_set=[("psnr", 2), ("lpips", 4), ("masked_lpips", 4),
                    ("boundary_lpips", 4)],
        caption=("Speed-matched comparison on " + dataset_name +
                 r", measured against the \emph{ground-truth} image "
                 r"(restoration quality). FreqSpec (full 50-step schedule) "
                 r"vs.\ reduced-step target baselines at comparable speedup. "
                 r"Boundary LPIPS (bLPIPS) is computed in a narrow band around "
                 r"the mask edge. Lower LPIPS is better."),
        label_suffix="speedmatch_gt",
        header_cols=[r"Speedup$\uparrow$", r"PSNR$\uparrow$",
                     r"LPIPS$\downarrow$", r"mLPIPS$\downarrow$",
                     r"bLPIPS$\downarrow$"],
    )

    out = gt_tab
    # (B) target-deviation table (only if available)
    if has_vs_tgt:
        td_tab = one_table(
            metric_set=[("lpips_vs_tgt", 4), ("masked_lpips_vs_tgt", 4),
                        ("boundary_lpips_vs_tgt", 4)],
            caption=("Target-fidelity comparison on " + dataset_name +
                     r", measured against the full-step target output "
                     r"(how closely each method reproduces the 50-step "
                     r"target). Lower is closer to the target."),
            label_suffix="speedmatch_tgt",
            header_cols=[r"Speedup$\uparrow$", r"LPIPS$_t\downarrow$",
                         r"mLPIPS$_t\downarrow$", r"bLPIPS$_t\downarrow$"],
        )
        out = gt_tab + "\n\n" + td_tab
    return out


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="Places2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--boundary_k", type=int, default=8,
                   help="Half-width (pixels) of the boundary band.")
    p.add_argument("--ref_steps", type=int, default=50,
                   help="Target step count used as 1.0x speedup reference.")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())