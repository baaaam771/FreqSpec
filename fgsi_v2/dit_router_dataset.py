#!/usr/bin/env python
"""
dit_router_dataset.py — Stage 6: teacher dataset for the draft-only router.

Rolls out DDIM trajectories advanced by the DENSE TARGET (the teacher
trajectory: the states the router will see are closest to target-quality
states), and at every collected timestep stores, per token:

    features : draft final hidden H_D (fp16), scalar eps_d stats, timestep
    label    : eps-space draft-target disagreement d_i = ||eps_d - eps_t||^2
               per token (fp32). Within one timestep this is perfectly rank-
               correlated with the x0 disagreement (they differ by a scalar
               1/alpha_bar factor), so eps-space is stored and the trainer
               rank-normalizes per (image, timestep).

Sharded output: <out_dir>/shard_XXXX.pt, each a dict of stacked tensors:
    h      [M, N, d_draft] fp16      scal [M, N, 7] fp16
    t      [M] long                  y    [M] long
    d_eps  [M, N] fp32
where M = trajectories x collected steps in the shard, N = tokens.

Usage (ImageNet-64):
    python dit_router_dataset.py \
        --target /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt --target_model DiT-S \
        --draft  /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --n_traj 2000 --steps 50 --t_stride 2 \
        --out_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/router_data
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit import count_params
from models.dit_sparse import dit_forward_tokens
from models.token_router import token_scalar_feats
from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    print(f"[router-data] target {count_params(target)/1e6:.1f}M | "
          f"draft {count_params(draft)/1e6:.1f}M")

    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    collect_steps = set(range(0, len(ts), args.t_stride))
    p = args.patch
    hw = (args.img_size // p, args.img_size // p)

    os.makedirs(args.out_dir, exist_ok=True)
    g = torch.Generator(device=dev).manual_seed(args.seed)

    buf = {k: [] for k in ("h", "scal", "t", "y", "d_eps")}
    shard_idx, rows_in_shard, total_rows = 0, 0, 0

    def flush():
        nonlocal shard_idx, rows_in_shard
        if not buf["h"]:
            return
        out = dict(h=torch.cat(buf["h"]).half().cpu(),
                   scal=torch.cat(buf["scal"]).half().cpu(),
                   t=torch.cat(buf["t"]).cpu(),
                   y=torch.cat(buf["y"]).cpu(),
                   d_eps=torch.cat(buf["d_eps"]).float().cpu())
        path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(out, path)
        print(f"[router-data] wrote {path} rows={out['h'].shape[0]}")
        for k in buf:
            buf[k].clear()
        shard_idx += 1
        rows_in_shard = 0

    n_done = 0
    while n_done < args.n_traj:
        b = min(args.batch, args.n_traj - n_done)
        y = torch.randint(0, args.num_classes, (b,), device=dev, generator=g)
        z = torch.randn(b, 3, args.img_size, args.img_size, device=dev, generator=g)
        for i, t in enumerate(ts):
            t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
            tt = torch.full((b,), int(t), device=dev, dtype=torch.long)
            eps_t = target(z, tt, y)
            if i in collect_steps:
                eps_d, h_d = dit_forward_tokens(draft, z, tt, y)
                d = torch.nn.functional.avg_pool2d(
                    (eps_d - eps_t).pow(2).mean(1, keepdim=True), p, stride=p)
                buf["h"].append(h_d)
                buf["scal"].append(token_scalar_feats(eps_d, p, hw))
                buf["t"].append(tt)
                buf["y"].append(y)
                buf["d_eps"].append(d.flatten(1))
                rows_in_shard += b
                total_rows += b
                if rows_in_shard >= args.shard_rows:
                    flush()
            # teacher trajectory: advance with the dense target
            z, _ = sch.ddim_step(z, eps_t, int(t), t_prev, eta=0.0)
        n_done += b
        print(f"[router-data] trajectories {n_done}/{args.n_traj} "
              f"(rows so far {total_rows})")
    flush()

    meta = dict(vars(args))
    meta.update(d_hidden_draft=int(draft.pos.shape[-1]),
                n_tokens=int(draft.num_tokens), rows=total_rows,
                ddim_ts=ts, collected_step_indices=sorted(collect_steps))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[router-data] done: {total_rows} rows in {shard_idx} shards -> {args.out_dir}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--n_traj", type=int, default=2000,
                    help="number of sampled trajectories")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--t_stride", type=int, default=2,
                    help="collect every t_stride-th DDIM step")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard_rows", type=int, default=8192,
                    help="rows per output shard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
