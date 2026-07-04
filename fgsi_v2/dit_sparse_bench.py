#!/usr/bin/env python
"""
dit_sparse_bench.py — Stage 10: does the FLOPs reduction survive as wall-clock?

Microbenchmarks ONE model forward (not a full trajectory):

    dense                  : full target forward (baseline)
    prefix+sparse_mlp      : dense prefix + hard-only-MLP suffix
    prefix+sparse_attn     : dense prefix + hard-query suffix
    draft                  : draft forward (for the total-system estimate)

over a grid of batch sizes, hard ratios, and split points, with CUDA-event
timing after warm-up (mean / median / p90 over --iters). Also reports the
analytic MAC ratio next to the measured latency ratio, so the gather/scatter
overhead is directly visible, plus peak memory.

At small token counts (256 tokens, DiT-S) the gather/scatter overhead can eat
the MLP savings at batch 1 — expect wall-clock wins to appear at batch >= 8 or
on larger grids; that is exactly what this script quantifies.

Usage:
    python dit_sparse_bench.py \
        --target ckpt/target.pt --target_model DiT-S \
        --draft ckpt/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --batches 1,4,8,16 --ratios 0.1,0.3,0.5,0.7 --splits 0.25,0.5,0.75 \
        --dtype bf16 --out results/dit_in64/sparse_bench.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit import count_params
from models.dit_sparse import (dit_forward_prefix, dit_forward_suffix_sparse,
                               dit_model_flops, topk_index)
from dit_token_sampler import load_dit


def timed(fn, dev, iters, warmup):
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ms = []
    for _ in range(iters):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0
    a = np.array(ms)
    return dict(mean_ms=round(float(a.mean()), 3),
                median_ms=round(float(np.median(a)), 3),
                p90_ms=round(float(np.percentile(a, 90)), 3),
                peak_mb=round(peak, 1))


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    dtype = dict(fp32=torch.float32, fp16=torch.float16,
                 bf16=torch.bfloat16)[args.dtype]
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    if dtype != torch.float32:
        target = target.to(dtype); draft = draft.to(dtype)
    L, N = len(target.blocks), target.num_tokens
    print(f"[bench] target {count_params(target)/1e6:.1f}M L={L} N={N} "
          f"dtype={args.dtype} | draft {count_params(draft)/1e6:.1f}M")

    batches = [int(x) for x in args.batches.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]
    splits = [float(x) for x in args.splits.split(",")]
    f_dense = dit_model_flops(target, "dense")

    rows = []
    for B in batches:
        x = torch.randn(B, 3, args.img_size, args.img_size, device=dev, dtype=dtype)
        t = torch.randint(0, 1000, (B,), device=dev)
        y = torch.randint(0, args.num_classes, (B,), device=dev)

        r_dense = timed(lambda: target(x, t, y), dev, args.iters, args.warmup)
        r_draft = timed(lambda: draft(x, t, y), dev, args.iters, args.warmup)
        rows.append(dict(kind="dense", batch=B, **r_dense, mac_ratio=1.0))
        rows.append(dict(kind="draft", batch=B, **r_draft,
                         mac_ratio=round(dit_model_flops(draft, "dense") / f_dense, 4)))
        print(f"[bench] B={B:3d} dense {r_dense['median_ms']:.2f}ms  "
              f"draft {r_draft['median_ms']:.2f}ms")

        for split in splits:
            m = max(0, min(L - 1, int(round(split * L))))
            for r in ratios:
                k = max(1, int(round(r * N)))
                idx = torch.sort(topk_index(torch.rand(B, N, device=dev), r),
                                 dim=1).values
                for mode in ("sparse_mlp", "sparse_attn"):
                    def run(mode=mode, m=m, idx=idx):
                        tok, c, hw = dit_forward_prefix(target, x, t, y, m)
                        dit_forward_suffix_sparse(target, tok, c, hw, idx, m, mode)
                    res = timed(run, dev, args.iters, args.warmup)
                    mac = dit_model_flops(target, mode, m=m, k=k) / f_dense
                    lat = res["median_ms"] / r_dense["median_ms"]
                    rows.append(dict(kind=mode, batch=B, split=split, ratio=r,
                                     m=m, k=k, mac_ratio=round(mac, 4),
                                     latency_ratio=round(lat, 4), **res))
                    print(f"[bench] B={B:3d} {mode:11s} m={m} r={r:.1f} "
                          f"mac={mac:.3f} lat={lat:.3f} "
                          f"({res['median_ms']:.2f}ms)")

    out = dict(target_model=args.target_model, draft_model=args.draft_model,
               img_size=args.img_size, tokens=N, blocks=L, dtype=args.dtype,
               iters=args.iters, rows=rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[bench] wrote {args.out}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--batches", type=str, default="1,4,8,16")
    ap.add_argument("--ratios", type=str, default="0.1,0.3,0.5,0.7")
    ap.add_argument("--splits", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--dtype", type=str, default="fp32",
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/sparse_bench.json")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
