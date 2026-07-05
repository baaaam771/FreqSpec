#!/usr/bin/env python
"""
dit_inpaint_eval.py — metrics from SAVED outputs (no resampling needed;
Stage-1 outputs can be re-scored with this script as long as images, masks,
and the gt list were stored).

Per run directory (samples/NNNNNN.png, masks/NNNNNN.png, gt_names.txt):
    mask LPIPS      : LPIPS( M.*out + (1-M).*gt , gt )   -- hole content
    boundary LPIPS  : same with the dilate-erode band mask
    known PSNR/SSIM : on the (1-M) region (reinjection sanity)
    full-image FID  : optional, clean-fid vs --ref_dir
    mask-size buckets: all of the above grouped by hole-area quartile

Usage:
    python dit_inpaint_eval.py \
        --run_dir results/inpaint/delta_mask_r0.3 \
        --data_dir /path/val_images [--ref_dir /path/val_images] \
        --out results/inpaint/delta_mask_r0.3/eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")))
    return x.permute(2, 0, 1).float() / 127.5 - 1.0


def band_of(M):
    dil = F.max_pool2d(M, 7, stride=1, padding=3)
    ero = -F.max_pool2d(-M, 7, stride=1, padding=3)
    return (dil - ero).clamp(0, 1)


def psnr(a, b, m):
    mse = ((a - b).pow(2) * m).sum() / (m.sum() * a.shape[0] + 1e-12)
    return float(10 * torch.log10(4.0 / (mse + 1e-12)))


def ssim_known(a, b, m):
    try:
        from skimage.metrics import structural_similarity as ssim
        an = ((a.permute(1, 2, 0).numpy() + 1) / 2)
        bn = ((b.permute(1, 2, 0).numpy() + 1) / 2)
        s, smap = ssim(an, bn, channel_axis=2, data_range=1.0, full=True)
        mn = m[0].numpy() > 0.5
        return float(smap[mn].mean()) if mn.any() else float(s)
    except Exception:
        return float("nan")


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    with open(os.path.join(args.run_dir, "gt_names.txt")) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    n = len(names)
    rows = []
    for i in range(0, n, args.batch):
        js = list(range(i, min(i + args.batch, n)))
        out = torch.stack([load(os.path.join(args.run_dir, "samples",
                                             f"{j:06d}.png")) for j in js])
        gt = torch.stack([load(os.path.join(args.data_dir, names[j]))
                          for j in js])
        if gt.shape[-1] != out.shape[-1]:
            gt = F.interpolate(gt, size=out.shape[-2:], mode="bicubic",
                               align_corners=False).clamp(-1, 1)
        M = torch.stack([torch.from_numpy(
            np.asarray(Image.open(os.path.join(args.run_dir, "masks",
                                               f"{j:06d}.png"))
                       ).astype(np.float32) / 255.0)[None]
            for j in js])
        out, gt, M = out.to(dev), gt.to(dev), M.to(dev)
        B = band_of(M)
        comp_m = M * out + (1 - M) * gt
        comp_b = B * out + (1 - B) * gt
        lm = lp(comp_m, gt).flatten()
        lb = lp(comp_b, gt).flatten()
        for k, j in enumerate(js):
            rows.append(dict(
                idx=j, area=float(M[k].mean()),
                mask_lpips=float(lm[k]), boundary_lpips=float(lb[k]),
                known_psnr=psnr(out[k].cpu(), gt[k].cpu(),
                                (1 - M[k]).cpu()),
                known_ssim=ssim_known(out[k].cpu(), gt[k].cpu(),
                                      (1 - M[k]).cpu())))
        print(f"[inp-eval] {min(i+args.batch, n)}/{n}")

    def agg(sub):
        return dict(n=len(sub),
                    mask_lpips=round(float(np.mean(
                        [r["mask_lpips"] for r in sub])), 4),
                    boundary_lpips=round(float(np.mean(
                        [r["boundary_lpips"] for r in sub])), 4),
                    known_psnr=round(float(np.mean(
                        [r["known_psnr"] for r in sub])), 2),
                    known_ssim=round(float(np.nanmean(
                        [r["known_ssim"] for r in sub])), 4))

    areas = np.array([r["area"] for r in rows])
    qs = np.quantile(areas, [0.25, 0.5, 0.75])
    buckets = {}
    lab = ["q1_small", "q2", "q3", "q4_large"]
    for bi in range(4):
        lo = -1 if bi == 0 else qs[bi - 1]
        hi = qs[bi] if bi < 3 else 2
        buckets[lab[bi]] = agg([r for r in rows if lo < r["area"] <= hi])

    out_json = dict(run=args.run_dir, overall=agg(rows),
                    by_mask_size=buckets,
                    area_quartiles=[round(float(q), 4) for q in qs])
    if args.ref_dir:
        from cleanfid import fid as cfid
        out_json["full_fid"] = round(float(cfid.compute_fid(
            os.path.join(args.run_dir, "samples"), args.ref_dir,
            mode="clean", num_workers=0)), 3)
    with open(args.out, "w") as f:
        json.dump(out_json, f, indent=2)
    print(json.dumps(out_json["overall"], indent=2))
    print(f"[inp-eval] wrote {args.out}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ref_dir", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
