#!/usr/bin/env python
"""
collect_sweep_metrics.py — Aggregate metrics from baseline_sweep.py runs.

For each given run dir and each requested FreqSpec method, computes:
  - wall-clock speedup vs target_s50 (paired per image, then averaged)
  - mean acceptance rate
  - LPIPS_t: LPIPS(method out, target_s50 out) — trajectory divergence,
    the paper's primary fidelity-to-target metric
  - target NFE mean

Works for cross-dataset runs, sensitivity runs, large-mask runs — any
directory produced by baseline_sweep.py. target_s50 may be a symlink.

Output: one CSV row per (run_dir, method) + pretty table on stdout.

Usage:
    python collect_sweep_metrics.py \\
        --run_dirs /mnt/.../cross_ffhq_draft-coco /mnt/.../cross_ffhq_draft-places2 \\
        --methods freqspec_strict freqspec_default \\
        --out_csv cross_metrics.csv \\
        --device cpu            # cpu is fine; cuda if GPU free

    # glob patterns work through the shell:
    python collect_sweep_metrics.py --run_dirs /mnt/.../cross_* --out_csv cross.csv
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_img_tensor(path, device):
    """PNG -> [1,3,H,W] in [-1,1]."""
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return (t * 2 - 1).to(device)


def read_results_csv(run_dir, method):
    """results.csv -> {idx: row_dict}."""
    path = Path(run_dir) / method / "results.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["idx"])] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["freqspec_strict", "freqspec_default"])
    p.add_argument("--ref_method", default="target_s50",
                   help="Reference for speedup + LPIPS_t.")
    p.add_argument("--out_csv", default="sweep_metrics.csv")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--max_images", type=int, default=0,
                   help="0 = all; set e.g. 50 to subsample for speed.")
    args = p.parse_args()

    import lpips
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(args.device).eval()

    rows_out = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.rstrip("/")
        run_name = os.path.basename(run_dir)
        ref_rows = read_results_csv(run_dir, args.ref_method)
        if not ref_rows:
            print(f"[collect] SKIP {run_name}: no {args.ref_method}/results.csv")
            continue

        for method in args.methods:
            m_rows = read_results_csv(run_dir, method)
            if not m_rows:
                print(f"[collect] SKIP {run_name}/{method}: no results.csv")
                continue

            common = sorted(set(ref_rows) & set(m_rows))
            if args.max_images > 0:
                common = common[:args.max_images]
            if not common:
                print(f"[collect] SKIP {run_name}/{method}: no common idx")
                continue

            speedups, accepts, lpips_t, tgt_nfe = [], [], [], []
            n_lpips_missing = 0
            with torch.no_grad():
                for idx in common:
                    r_ref = ref_rows[idx]
                    r_m = m_rows[idx]
                    t_ref = float(r_ref["time_sec"])
                    t_m = float(r_m["time_sec"])
                    if t_m > 0:
                        speedups.append(t_ref / t_m)
                    if r_m.get("accept_rate"):
                        accepts.append(float(r_m["accept_rate"]))
                    if r_m.get("target_nfe"):
                        tgt_nfe.append(float(r_m["target_nfe"]))
                    # LPIPS_t vs reference output
                    p_ref = Path(run_dir) / args.ref_method / f"img_{idx:03d}" / "out.png"
                    p_m = Path(run_dir) / method / f"img_{idx:03d}" / "out.png"
                    if p_ref.exists() and p_m.exists():
                        a = load_img_tensor(p_ref, args.device)
                        b = load_img_tensor(p_m, args.device)
                        d = loss_fn(a, b).item()
                        lpips_t.append(d)
                    else:
                        n_lpips_missing += 1

            row = {
                "run": run_name,
                "method": method,
                "n": len(common),
                "speedup_mean": np.mean(speedups) if speedups else float("nan"),
                "speedup_std": np.std(speedups) if speedups else float("nan"),
                "accept_mean": np.mean(accepts) if accepts else float("nan"),
                "lpips_t_mean": np.mean(lpips_t) if lpips_t else float("nan"),
                "lpips_t_std": np.std(lpips_t) if lpips_t else float("nan"),
                "target_nfe_mean": np.mean(tgt_nfe) if tgt_nfe else float("nan"),
            }
            rows_out.append(row)
            print(f"[collect] {run_name:40s} {method:18s} "
                  f"n={row['n']:3d}  spd={row['speedup_mean']:.3f}  "
                  f"acc={row['accept_mean']:.3f}  "
                  f"LPIPS_t={row['lpips_t_mean']:.4f}"
                  + (f"  ({n_lpips_missing} imgs missing)" if n_lpips_missing else ""))

    if rows_out:
        keys = list(rows_out[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n[collect] wrote {len(rows_out)} rows -> {args.out_csv}")
    else:
        print("[collect] nothing collected — check paths")


if __name__ == "__main__":
    main()
