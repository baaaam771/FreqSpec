#!/usr/bin/env python
"""
dit_imagenet_sanity.py -- Phase 0 sanity check for the ImageNet-64 pipeline.

Verifies, before any training, that:
  - the dataset loads with class labels (class-conditional),
  - images are [B,3,H,W] normalized to [-1,1],
  - labels are in range [0, num_classes-1],
  - a DiT forward pass runs and produces eps in [B,3,H,W],
  - patchification yields the expected token count (e.g. 64x64 patch4 -> 256).

Usage:
    python dit_imagenet_sanity.py \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/imagenet64/train \
        --img_size 64 --patch 4 --num_classes 1000 --model DiT-S
"""
import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.dit import build_dit, count_params


def main(a):
    dev = a.device
    tf = transforms.Compose([transforms.Resize(a.img_size),
                             transforms.CenterCrop(a.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.ImageFolder(a.data_root, transform=tf)
    print(f"[sanity] dataset: {len(ds)} images, {len(ds.classes)} classes")
    assert len(ds.classes) <= a.num_classes, \
        f"found {len(ds.classes)} classes > --num_classes {a.num_classes}"

    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True)
    x, y = next(iter(dl))
    print(f"[sanity] batch image shape: {tuple(x.shape)}  dtype={x.dtype}")
    print(f"[sanity] batch label shape: {tuple(y.shape)}  range=[{int(y.min())},{int(y.max())}]")
    print(f"[sanity] pixel range: [{x.min():.3f}, {x.max():.3f}] (expect ~[-1,1])")

    ok = True
    if tuple(x.shape) != (a.batch, 3, a.img_size, a.img_size):
        print("  !! image shape mismatch"); ok = False
    if int(y.max()) >= a.num_classes:
        print("  !! label out of range"); ok = False
    if x.min() < -1.05 or x.max() > 1.05:
        print("  !! pixel range outside [-1,1]"); ok = False

    # DiT forward
    m = build_dit(a.model, img_size=a.img_size, patch=a.patch,
                  num_classes=a.num_classes).to(dev).eval()
    print(f"[sanity] {a.model}: {count_params(m)/1e6:.1f}M params, num_tokens={m.num_tokens}")
    exp_tokens = (a.img_size // a.patch) ** 2
    if m.num_tokens != exp_tokens:
        print(f"  !! token count {m.num_tokens} != expected {exp_tokens}"); ok = False

    with torch.no_grad():
        xb = x[: min(a.batch, 4)].to(dev)
        yb = y[: min(a.batch, 4)].to(dev)
        t = torch.randint(0, 1000, (xb.shape[0],), device=dev)
        eps = m(xb, t, yb)
    print(f"[sanity] DiT forward eps shape: {tuple(eps.shape)} (expect {tuple(xb.shape)})")
    if tuple(eps.shape) != tuple(xb.shape):
        print("  !! eps shape mismatch"); ok = False

    # patchify check (avg_pool to token grid)
    pooled = F.avg_pool2d(eps, a.patch, stride=a.patch)
    grid = a.img_size // a.patch
    print(f"[sanity] token grid after patchify: {tuple(pooled.shape[-2:])} = {grid*grid} tokens")
    if tuple(pooled.shape[-2:]) != (grid, grid):
        print("  !! patchify grid mismatch"); ok = False

    print("\n[sanity] RESULT:", "PASS \u2713 -> proceed to Phase 1 (target/draft training)"
          if ok else "FAIL \u2717 -> fix data/config before training")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--model", type=str, default="DiT-S")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
