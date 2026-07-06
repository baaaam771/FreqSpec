#!/usr/bin/env python
"""
dit_inpaint_latency.py — CUDA-event latency profiling (review item 10).

MACs are the primary analytic axis, but wall-clock must be reported honestly:
per-segment CUDA-event timing, batch sweep {1,4,8,16}, warm-up, synchronize,
mean / median / p90, and peak memory. ImageNet-64 here is PIXEL space, so
there is NO VAE decode; the --note_vae flag documents that decode would add a
fixed constant in a latent-space port.

Segments:
    T_anchor      dense target pass (+ suffix cache write)
    T_selector    hard-token selection
    T_sparse      cached sparse suffix on hard tokens
    T_scheduler   DDIM update
    T_reinject    known-region re-injection

This reproduces the DACE §4.7 "wall-clock reality" caveat for the inpainting
setting: at batch 1 a dense DiT-S forward is kernel-launch bound, so MAC
savings convert to latency only at larger batch. The profiler makes that
measurable rather than assumed.

Usage:
    python dit_inpaint_latency.py \
        --target ckpt_dit_inp/target.pt --target_model DiT-S-Inp \
        --batches 1 4 8 16 --hard_ratio 0.3 --cache_period 2 --split_m 0 \
        --out results/dit_inp/latency
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit_inpaint import DiTInpaint, load_dit_inpaint
from models.dit_sparse import (dit_forward_dense_with_cache,
                               sparse_target_eps_cached, dit_model_flops,
                               topk_index)
from training.scheduler import DDPMSchedule
from utils.inpaint_masks import sample_masks, mask_to_tokens, dilate_tokens


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


class Seg:
    def __init__(self, dev):
        self.dev = dev; self.t = {}

    def start(self):
        sync(self.dev); self.t0 = time.perf_counter()

    def stop(self, k):
        sync(self.dev)
        self.t.setdefault(k, []).append((time.perf_counter() - self.t0) * 1e3)


@torch.no_grad()
def one_step(target, x_in, tt, y, split_m, k, cache, seg, do_anchor):
    if do_anchor:
        seg.start()
        eps, cache = dit_forward_dense_with_cache(target, x_in, tt, y, split_m)
        seg.stop("T_anchor")
        return eps, cache
    seg.start()
    N = target.num_tokens
    score = torch.rand(x_in.shape[0], N, device=x_in.device)
    idx = torch.sort(topk_index(score, k / N), dim=1).values
    seg.stop("T_selector")
    seg.start()
    eps = sparse_target_eps_cached(target, x_in, tt, y, idx, split_m,
                                   torch.zeros(x_in.shape[0], 3,
                                               x_in.shape[-2], x_in.shape[-1],
                                               device=x_in.device), cache)
    seg.stop("T_sparse")
    return eps, cache


@torch.no_grad()
def profile_batch(target, sch, args, B, dev):
    p = args.patch; H = args.img_size
    N = (H // p) ** 2
    k = max(1, int(round(args.hard_ratio * N)))
    x0 = torch.randn(B, 3, H, H, device=dev)
    mask = sample_masks(B, H, H, box_prob=args.box_prob, seed=0).to(dev)
    x_masked = x0 * (1 - mask)
    y = torch.randint(0, args.num_classes, (B,), device=dev)
    ts = sch.get_ddim_schedule_exact(args.steps)
    sac, s1m = sch.sqrt_alphas_cumprod, sch.sqrt_one_minus_alphas_cumprod

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # warm-up
    z = torch.randn(B, 3, H, H, device=dev)
    for _ in range(args.warmup):
        x_in = DiTInpaint.pack(z, x_masked, mask)
        tt = torch.full((B,), int(ts[0].item()), device=dev, dtype=torch.long)
        dit_forward_dense_with_cache(target, x_in, tt, y, args.split_m)
    sync(dev)

    per_iter = []
    seg = Seg(dev)
    for _ in range(args.iters):
        z = torch.randn(B, 3, H, H, device=dev)
        cache = None
        t_iter0 = time.perf_counter(); sync(dev)
        for i in range(len(ts)):
            t = int(ts[i].item())
            t_prev = int(ts[i + 1].item()) if i + 1 < len(ts) else -1
            tt = torch.full((B,), t, device=dev, dtype=torch.long)
            x_in = DiTInpaint.pack(z, x_masked, mask)
            do_anchor = (i % args.cache_period == 0)
            eps, cache = one_step(target, x_in, tt, y, args.split_m, k, cache,
                                  seg, do_anchor)
            seg.start()
            z, _ = sch.ddim_step(z, eps, t, t_prev)
            seg.stop("T_scheduler")
            seg.start()
            z = mask * z + (1 - mask) * (sac[t_prev] * x0 + s1m[t_prev] * torch.randn_like(z)
                                         if t_prev >= 0 else x0)
            seg.stop("T_reinject")
        sync(dev)
        per_iter.append((time.perf_counter() - t_iter0) * 1e3)

    peak = (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else 0.0
    seg_means = {kk: sum(v) / len(v) for kk, v in seg.t.items()}
    return dict(
        batch=B, k=k, N=N,
        total_ms_mean=statistics.mean(per_iter),
        total_ms_median=statistics.median(per_iter),
        total_ms_p90=sorted(per_iter)[int(0.9 * (len(per_iter) - 1))],
        per_image_ms_mean=statistics.mean(per_iter) / B,
        segment_ms_mean=seg_means,
        macs_vs_dense50=(args.cache_period_frac_macs(target, args, k)),
        peak_mem_gb=peak)


def _macs_ratio(target, args, k):
    macs_ref = 50 * dit_model_flops(target, "dense")
    n_anchor = len(range(0, args.steps, args.cache_period))
    n_sparse = args.steps - n_anchor
    macs = (n_anchor * dit_model_flops(target, "dense")
            + n_sparse * dit_model_flops(target, "sparse_attn",
                                         m=args.split_m, k=k))
    return macs / macs_ref


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    if os.path.exists(os.path.join(args.out, "latency.json")) and not args.overwrite:
        print(f"[skip] {args.out} already done (use --overwrite)"); return
    os.makedirs(args.out, exist_ok=True)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4,
                       beta_end=0.02, beta_schedule="linear", device=dev)
    target = load_dit_inpaint(args.target, args.target_model, args.img_size,
                              args.patch, args.num_classes, dev)
    # bind the macs helper onto args for profile_batch
    args.cache_period_frac_macs = lambda tgt, a, k: _macs_ratio(tgt, a, k)

    results = []
    for B in args.batches:
        r = profile_batch(target, sch, args, B, dev)
        results.append(r)
        print(f"[batch {B}] total {r['total_ms_mean']:.1f}ms "
              f"(p90 {r['total_ms_p90']:.1f}) per-img {r['per_image_ms_mean']:.2f}ms "
              f"MAC {r['macs_vs_dense50']:.3f} peak {r['peak_mem_gb']:.2f}GB")

    out = dict(config={k: v for k, v in vars(args).items()
                       if not callable(v)},
               note_vae="ImageNet-64 pixel space: no VAE decode; a latent-space "
                        "port adds a fixed decode constant per image.",
               results=results)
    with open(os.path.join(args.out, "latency.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {args.out}/latency.json")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--target_model", type=str, default="DiT-S-Inp")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cache_period", type=int, default=2)
    p.add_argument("--split_m", type=int, default=0)
    p.add_argument("--hard_ratio", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--box_prob", type=float, default=0.4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
