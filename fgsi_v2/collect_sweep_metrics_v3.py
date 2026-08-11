#!/usr/bin/env python
"""
collect_sweep_metrics_v3.py — Per-image + tail-aware metric aggregation,
now with GT-based quality metrics (addresses "both models can be wrong
together"): masked-region LPIPS vs the ORIGINAL image and boundary-band
LPIPS (16px band around the mask edge), computed via spatial LPIPS maps.
LPIPS_t (divergence vs target_s50) remains as the trajectory-fidelity
auxiliary metric. The GT image is auto-detected per img dir among
{gt.png, in.png, input.png, original.png}; if none exists, GT columns are
NaN and a warning names the files actually present.

FAILURE THRESHOLD PROTOCOL: do NOT tune --fail_thr on the final test
numbers. Split per-image rows into a calibration half (e.g. even idx) and
a test half; choose the threshold on calibration (e.g. p95 of the
reference operating point), then report the test half only.

Extends v1 with everything needed for the tail/failure-rate positioning:
  PER-IMAGE CSV  : one row per (run, method, idx) — speedup, accept,
                   LPIPS_t, target_nfe, time_sec, mask coverage, mask
                   boundary complexity. This is the raw log for
                   difficulty-stratified analysis and budget-policy
                   simulation.
  SUMMARY CSV    : per (run, method): mean/std of the above, PLUS
                   p50/p90/p95 latency, worst-10% LPIPS_t mean (CVaR),
                   failure rate (LPIPS_t > --fail_thr).

Mask statistics (from the saved mask.png, deterministic per idx):
  coverage    = fraction of masked pixels
  complexity  = perimeter / (2*sqrt(pi*area))   (isoperimetric ratio;
                1.0 = disk, larger = more irregular/brushy boundary)
Use these to PRE-DEFINE difficulty strata (small/large x regular/irregular)
instead of post-hoc selection by acceptance — avoids selection bias.

Usage:
    python collect_sweep_metrics_v2.py \\
        --run_dirs /mnt/.../main_coco_400k /mnt/.../main_ffhq_400k \\
        --methods freqspec_strict freqspec_mid freqspec_default \\
        --ref_method target_s50 \\
        --out_prefix phase2 --device cpu
    # -> phase2_per_image.csv, phase2_summary.csv
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_img_tensor(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return (t * 2 - 1).to(device)


def mask_stats(mask_path):
    """coverage + isoperimetric boundary complexity from mask.png."""
    m = (np.array(Image.open(mask_path).convert("L")) > 127)
    area = float(m.sum())
    H, W = m.shape
    coverage = area / (H * W)
    if area == 0:
        return coverage, 0.0
    # boundary pixels: masked pixels with at least one unmasked 4-neighbor
    pad = np.pad(m, 1, mode="edge")
    interior = (pad[1:-1, :-2] & pad[1:-1, 2:] &
                pad[:-2, 1:-1] & pad[2:, 1:-1])
    boundary = m & ~interior
    perim = float(boundary.sum())
    complexity = perim / (2.0 * np.sqrt(np.pi * area))
    return coverage, complexity


def read_results_csv(run_dir, method):
    path = Path(run_dir) / method / "results.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["idx"])] = row
    return out


GT_CANDIDATES = ["gt.png", "in.png", "input.png", "original.png"]


def find_gt(img_dir):
    for name in GT_CANDIDATES:
        p = img_dir / name
        if p.exists():
            return p
    return None


def load_mask_bool(path):
    import numpy as _np
    m = _np.array(Image.open(path).convert("L")) > 127
    return torch.from_numpy(m)


def boundary_band(mask_bool, k=16):
    """band = dilate(mask,k) XOR erode(mask,k) as float [1,1,H,W]."""
    import torch.nn.functional as F
    m = mask_bool.float().unsqueeze(0).unsqueeze(0)
    pad = k // 2
    dil = F.max_pool2d(m, kernel_size=k | 1, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=k | 1, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def region_lpips(loss_fn_spatial, a, b, region):
    """Mean of the spatial LPIPS map over region ([1,1,h,w] weights)."""
    import torch.nn.functional as F
    d_map = loss_fn_spatial(a, b)          # [1,1,H',W']
    r = F.interpolate(region, size=d_map.shape[-2:], mode="nearest")
    denom = r.sum().item()
    if denom < 1:
        return float("nan")
    return ((d_map * r).sum() / denom).item()


def pctl(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["freqspec_strict", "freqspec_mid",
                            "freqspec_default"])
    p.add_argument("--ref_method", default="target_s50")
    p.add_argument("--out_prefix", default="metrics")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--fail_thr", type=float, default=0.08,
                   help="LPIPS_t above this counts as a failure. Calibrate "
                        "on the observed distribution (e.g. p95 of the "
                        "reference operating point) before quoting.")
    p.add_argument("--max_images", type=int, default=0)
    args = p.parse_args()

    import lpips
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(args.device).eval()
    loss_fn_sp = lpips.LPIPS(net=args.lpips_net, spatial=True).to(args.device).eval()
    gt_warned = set()

    per_image, summary = [], []
    mask_cache = {}  # (run_dir, idx) -> (coverage, complexity)

    for run_dir in args.run_dirs:
        run_dir = run_dir.rstrip("/")
        run_name = os.path.basename(run_dir)
        ref_rows = read_results_csv(run_dir, args.ref_method)
        if not ref_rows:
            print(f"[v2] SKIP {run_name}: no {args.ref_method}")
            continue

        for method in args.methods:
            m_rows = read_results_csv(run_dir, method)
            common = sorted(set(ref_rows) & set(m_rows))
            if args.max_images > 0:
                common = common[:args.max_images]
            if not common:
                print(f"[v2] SKIP {run_name}/{method}: no common idx")
                continue

            sp, ac, lt, nfe, tsec = [], [], [], [], []
            gml, gbl = [], []
            with torch.no_grad():
                for idx in common:
                    rr, rm = ref_rows[idx], m_rows[idx]
                    t_ref, t_m = float(rr["time_sec"]), float(rm["time_sec"])
                    spd = t_ref / t_m if t_m > 0 else float("nan")
                    acc = float(rm["accept_rate"]) if rm.get("accept_rate") else float("nan")
                    tn = float(rm["target_nfe"]) if rm.get("target_nfe") else float("nan")

                    p_ref = Path(run_dir) / args.ref_method / f"img_{idx:03d}" / "out.png"
                    p_m = Path(run_dir) / method / f"img_{idx:03d}" / "out.png"
                    d = float("nan")
                    if p_ref.exists() and p_m.exists():
                        d = loss_fn(load_img_tensor(p_ref, args.device),
                                    load_img_tensor(p_m, args.device)).item()

                    key = (run_dir, idx)
                    if key not in mask_cache:
                        mp = Path(run_dir) / method / f"img_{idx:03d}" / "mask.png"
                        mask_cache[key] = mask_stats(mp) if mp.exists() else (float("nan"),) * 2
                    cov, cpx = mask_cache[key]

                    # GT-based quality: masked-region + boundary-band LPIPS
                    img_dir = Path(run_dir) / method / f"img_{idx:03d}"
                    gt_path = find_gt(img_dir)
                    mp2 = img_dir / "mask.png"
                    gt_ml, gt_bl = float("nan"), float("nan")
                    if gt_path is not None and mp2.exists() and p_m.exists():
                        g = load_img_tensor(gt_path, args.device)
                        o = load_img_tensor(p_m, args.device)
                        mb = load_mask_bool(mp2)
                        region_m = mb.float().unsqueeze(0).unsqueeze(0).to(args.device)
                        region_b = boundary_band(mb).to(args.device)
                        gt_ml = region_lpips(loss_fn_sp, g, o, region_m)
                        gt_bl = region_lpips(loss_fn_sp, g, o, region_b)
                    elif gt_path is None and (run_dir, method) not in gt_warned:
                        gt_warned.add((run_dir, method))
                        present = sorted(p.name for p in img_dir.glob("*.png"))
                        print(f"[v3] WARN {run_name}/{method}: no GT image "
                              f"among {GT_CANDIDATES}; files present: {present} "
                              f"-> gt_* columns will be NaN")

                    per_image.append({
                        "run": run_name, "method": method, "idx": idx,
                        "time_sec": t_m, "speedup": spd, "accept": acc,
                        "lpips_t": d, "target_nfe": tn,
                        "gt_masked_lpips": gt_ml,
                        "gt_boundary_lpips": gt_bl,
                        "mask_coverage": round(cov, 4),
                        "mask_complexity": round(cpx, 3),
                    })
                    sp.append(spd); ac.append(acc); lt.append(d)
                    nfe.append(tn); tsec.append(t_m)
                    gml.append(gt_ml); gbl.append(gt_bl)

            lt_valid = sorted(x for x in lt if not np.isnan(x))
            k10 = max(1, len(lt_valid) // 10)
            worst10 = float(np.mean(lt_valid[-k10:])) if lt_valid else float("nan")
            fails = sum(1 for x in lt_valid if x > args.fail_thr)

            row = {
                "run": run_name, "method": method, "n": len(common),
                "speedup_mean": float(np.nanmean(sp)),
                "accept_mean": float(np.nanmean(ac)),
                "lpips_t_mean": float(np.nanmean(lt)),
                "lpips_t_p90": pctl(lt_valid, 90),
                "lpips_t_worst10_mean": worst10,
                "fail_rate": fails / max(1, len(lt_valid)),
                "time_p50": pctl(tsec, 50),
                "time_p90": pctl(tsec, 90),
                "time_p95": pctl(tsec, 95),
                "target_nfe_mean": float(np.nanmean(nfe)),
                "gt_masked_lpips_mean": float(np.nanmean(gml)) if gml else float("nan"),
                "gt_masked_lpips_worst10": (lambda v: float(np.mean(sorted(v)[-max(1, len(v)//10):])) if v else float("nan"))([x for x in gml if not np.isnan(x)]),
                "gt_boundary_lpips_mean": float(np.nanmean(gbl)) if gbl else float("nan"),
            }
            summary.append(row)
            print(f"[v2] {run_name:28s} {method:18s} n={row['n']:3d} "
                  f"spd={row['speedup_mean']:.3f} acc={row['accept_mean']:.3f} "
                  f"LPIPS_t={row['lpips_t_mean']:.4f} "
                  f"worst10={row['lpips_t_worst10_mean']:.4f} "
                  f"fail={row['fail_rate']:.3f}")

    for name, rows in [("per_image", per_image), ("summary", summary)]:
        if not rows:
            continue
        path = f"{args.out_prefix}_{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[v2] wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
