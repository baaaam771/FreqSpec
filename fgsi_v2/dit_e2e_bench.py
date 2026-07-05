#!/usr/bin/env python
"""
dit_e2e_bench.py — end-to-end wall-clock of FULL sampling runs.

TOTAL counts model MACs only; r=0 anchored reuse performs MORE scheduler
updates than its dense counterpart at equal target evaluations (e.g. dense
s10: 10 fwd + 10 updates vs. r=0 S=50/c=5: 10 fwd + 50 updates). This script
measures whether the FID win survives in real time, under matched
conditions: same batch, same dtype policy, warm-up run excluded, CUDA
synchronize around the whole sampling loop, no image saving / FID.

Reports per config: wall s/run, s/image, ms/step, and the exact call counts
(target dense / sparse, draft, scheduler updates) from the sampler's own
counters, so the eval/update accounting is verifiable per row.

Usage:
    python dit_e2e_bench.py \
        --target ckpt/target.pt --target_model DiT-S \
        --draft ckpt/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --batch 128 --repeats 5 --dtype bf16 \
        --out results/dit_in64_sparse/e2e_wallclock.json
Default config set: dense s10/s15, r=0 {s20cp2, s30cp2, s50cp5, s50cp3};
override with repeated --config kind,steps[,cp] (kind in {dense,r0}).
"""
import argparse
import contextlib
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit
from dit_sparse_sampler import sample_sparse

DEFAULT_CONFIGS = ["dense,10", "dense,15", "r0,20,2", "r0,30,2",
                   "r0,50,5", "r0,50,3"]


def make_args(kind, steps, cp, base):
    a = Namespace(**vars(base))
    a.steps = steps
    if kind == "dense":
        a.selector, a.suffix_mode = "dense", "cache_attn"
        a.hard_ratio, a.cache_period = 0.3, 5
    else:  # r0
        a.selector, a.suffix_mode = "random", "cache_attn"
        a.easy_source, a.hard_ratio = "target_cache", 0.0
        a.cache_period = cp
    return a


@torch.no_grad()
def run_once(target, draft, sch, cfg_args, y, z, dev, dtype):
    ts = sch.get_ddim_schedule_exact(cfg_args.steps).tolist()
    ctxm = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (dtype == "bf16" and dev.type == "cuda")
            else contextlib.nullcontext())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with ctxm:
        _, _, n_warm, n_anchor, cnt = sample_sparse(
            target, draft, sch, ts, y, z.clone(), cfg_args, dev, None)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, cnt


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4,
                       beta_end=0.02, beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)

    base = Namespace(selector="dense", suffix_mode="cache_attn",
                     hard_ratio=0.0, split=0.0, cache_period=5,
                     refresh_every=0, dense_until=1.0,
                     easy_source="target_cache", patch=args.patch,
                     steps=50)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    y = torch.randint(0, args.num_classes, (args.batch,), device=dev,
                      generator=g)
    z = torch.randn(args.batch, 3, args.img_size, args.img_size,
                    device=dev, generator=g)

    rows = []
    for spec in (args.config or DEFAULT_CONFIGS):
        f = spec.split(",")
        kind, steps = f[0].strip(), int(f[1])
        cp = int(f[2]) if len(f) > 2 else 1
        ca = make_args(kind, steps, cp, base)
        run_once(target, draft, sch, ca, y, z, dev, args.dtype)  # warm-up
        secs = []
        for _ in range(args.repeats):
            s, cnt = run_once(target, draft, sch, ca, y, z, dev, args.dtype)
            secs.append(s)
        a = np.array(secs)
        name = f"{kind}_s{steps}" + (f"_cp{cp}" if kind == "r0" else "")
        row = dict(config=name, kind=kind, steps=steps, cp=cp,
                   target_evals=cnt["target_dense"],
                   sparse_evals=cnt["target_sparse"],
                   draft_evals=cnt["draft"],
                   scheduler_updates=steps,
                   wall_s_median=round(float(np.median(a)), 4),
                   wall_s_mean=round(float(a.mean()), 4),
                   wall_s_std=round(float(a.std()), 4),
                   s_per_image=round(float(np.median(a)) / args.batch, 5),
                   ms_per_step=round(float(np.median(a)) / steps * 1e3, 2))
        rows.append(row)
        print(f"[e2e] {name:16s} evals={row['target_evals']:3d} "
              f"updates={steps:3d} wall={row['wall_s_median']:.3f}s "
              f"({row['s_per_image']*1e3:.1f} ms/img)")

    dense_ref = {r["config"]: r["wall_s_median"] for r in rows
                 if r["kind"] == "dense"}
    for r in rows:
        if r["kind"] == "r0":
            peer = dense_ref.get(f"dense_s{r['target_evals']}")
            if peer:
                r["wall_vs_dense_same_evals"] = round(
                    r["wall_s_median"] / peer, 3)

    out = dict(batch=args.batch, dtype=args.dtype, repeats=args.repeats,
               device=str(dev),
               gpu=(torch.cuda.get_device_name(0)
                    if dev.type == "cuda" else "cpu"),
               rows=rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[e2e] wrote {args.out}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", type=str, default="bf16",
                    choices=["fp32", "bf16"],
                    help="bf16 uses autocast (scheduler math stays fp32)")
    ap.add_argument("--config", action="append", default=None,
                    help="kind,steps[,cp] -- repeatable; kind in {dense,r0}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/e2e_wallclock.json")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
