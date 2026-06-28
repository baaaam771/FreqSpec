#!/usr/bin/env python
"""
dit_quality_gate.py -- Phase 1 gate: sample target-only and draft-only grids
to confirm both models are usable before the reliability/mixing experiments.

Loads a trained target and (optionally) draft DiT, runs plain DDIM sampling for
each alone, saves sample grids, and reports pixel-std as a collapse proxy. The
gate: target must NOT collapse (recognizable, std comparable to data), draft is
weaker but not pure noise (or at least recoverable by token mixing).

Usage:
    python dit_quality_gate.py \
        --target /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt --target_model DiT-S \
        --draft  /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --out_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/quality_gate \
        --n_samples 64 --steps 50
"""
import argparse
import os

import numpy as np
import torch
import torchvision.utils as vutils

from models.dit import build_dit
from training.scheduler import DDPMSchedule


def load_dit(path, name, a, dev):
    ck = torch.load(path, map_location=dev)
    sd = ck.get("ema", ck.get("model", ck))
    m = build_dit(name, img_size=a.img_size, patch=a.patch, num_classes=a.num_classes)
    m.load_state_dict(sd); m.to(dev).eval()
    return m


@torch.no_grad()
def ddim_sample(model, a, dev, sch):
    n = a.n_samples
    x = torch.randn(n, 3, a.img_size, a.img_size, device=dev)
    y = torch.randint(0, a.num_classes, (n,), device=dev)
    ts = sch.get_ddim_schedule_exact(a.steps).tolist()
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        eps = model(x, torch.full((n,), int(t), device=dev, dtype=torch.long), y)
        x, _ = sch.ddim_step(x, eps, int(t), int(t_prev), eta=0.0)
    return x.clamp(-1, 1)


def save_grid(x, path, nrow=8):
    g = vutils.make_grid((x + 1) / 2, nrow=nrow, padding=2)
    vutils.save_image(g, path)


def main(a):
    dev = a.device
    os.makedirs(a.out_dir, exist_ok=True)
    sch = DDPMSchedule(device=dev)
    report = {}

    tgt = load_dit(a.target, a.target_model, a, dev)
    xt = ddim_sample(tgt, a, dev, sch)
    save_grid(xt, os.path.join(a.out_dir, "grid_target.png"))
    report["target"] = float(xt.std())
    print(f"[gate] target-only px-std={report['target']:.3f}  "
          f"-> grid_target.png")

    if a.draft:
        dft = load_dit(a.draft, a.draft_model, a, dev)
        xd = ddim_sample(dft, a, dev, sch)
        save_grid(xd, os.path.join(a.out_dir, "grid_draft.png"))
        report["draft"] = float(xd.std())
        print(f"[gate] draft-only  px-std={report['draft']:.3f}  "
              f"-> grid_draft.png")

    import json
    json.dump(report, open(os.path.join(a.out_dir, "quality_gate.json"), "w"), indent=2)

    # gate verdict (heuristic; final call is visual)
    ts = report["target"]
    msg = []
    if 0.10 < ts < 0.60:
        msg.append("target std in a plausible range")
    else:
        msg.append(f"WARN: target std {ts:.3f} unusual (check collapse/over-noise)")
    if "draft" in report:
        ds = report["draft"]
        if ds < ts:
            msg.append(f"draft weaker than target (std {ds:.3f} < {ts:.3f}) -- good gap")
        else:
            msg.append(f"WARN: draft std {ds:.3f} >= target {ts:.3f} (gap may be too small)")
    print("[gate] " + "; ".join(msg))
    print("[gate] Inspect grids visually: target must be recognizable, "
          "draft weaker but not pure noise.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--target_model", type=str, default="DiT-S")
    p.add_argument("--draft", type=str, default="")
    p.add_argument("--draft_model", type=str, default="DiT-Nano")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
