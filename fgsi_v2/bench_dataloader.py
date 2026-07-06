#!/usr/bin/env python
"""
bench_dataloader.py — HDD에서 --workers 몇이 최적인지 빠르게 측정.

128만 개 PNG를 HDD에서 random read 할 때 GPU가 굶는 문제를 진단/튜닝하기 위한
도구. 학습 없이 dataloader 처리량(images/sec)만 측정한다. GPU 연산은 짧게
흉내내서(옵션) I/O가 진짜 병목인지 확인.

Usage:
    python bench_dataloader.py \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/imagenet64/train \
        --batch 256 --workers 0 2 4 8 12 --n_batches 40
"""
import argparse
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "training"
                else os.path.dirname(os.path.abspath(__file__)))


def _worker_init(_wid):
    import random
    import numpy as np
    s = torch.initial_seed() % (2 ** 31)
    np.random.seed(s); random.seed(s)


def build_loader(args, nw):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize(args.img_size),
                             transforms.CenterCrop(args.img_size),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    kw = dict(batch_size=args.batch, shuffle=True, drop_last=True,
              num_workers=nw, pin_memory=True)
    if nw > 0:
        kw.update(persistent_workers=True, prefetch_factor=args.prefetch_factor,
                  worker_init_fn=_worker_init)
    return torch.utils.data.DataLoader(ds, **kw), len(ds)


def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  batch={args.batch}  n_batches={args.n_batches}  "
          f"prefetch={args.prefetch_factor}", flush=True)
    for nw in args.workers:
        loader, n = build_loader(args, nw)
        it = iter(loader)
        # warm-up (worker spin-up / first reads)
        for _ in range(args.warmup):
            next(it)
        t0 = time.perf_counter(); seen = 0
        for _ in range(args.n_batches):
            x, y = next(it)
            x = x.to(dev, non_blocking=True)
            if args.fake_compute:
                # 짧은 GPU 연산으로 I/O 겹침 여부 확인
                (x * 1.0001).sum()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
            seen += x.shape[0]
        dt = time.perf_counter() - t0
        ips = seen / dt
        print(f"workers={nw:2d}  {ips:8.1f} img/s  "
              f"({dt:.2f}s for {seen} imgs)  "
              f"→ 300k steps ≈ {args.batch*300000/ips/3600:.1f} h",
              flush=True)
        del loader, it


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prefetch_factor", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--fake_compute", action="store_true", default=True)
    main(ap.parse_args())
