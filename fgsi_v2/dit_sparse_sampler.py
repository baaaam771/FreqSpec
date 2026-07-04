#!/usr/bin/env python
"""
dit_sparse_sampler.py — sparse target execution on the DiT token grid.

Extends the K=1 token-mixing sampler (dit_token_sampler.py) from OUTPUT mixing
(both models run densely) to actual SPARSE TARGET EXECUTION:

    1. draft full forward              eps_d, H_D
    2. hard-token selection            idx = TopK(score, r)   [selector below]
    3. target dense prefix             blocks 0..m-1, all tokens
    4. target sparse suffix            blocks m..L-1, hard tokens only
       ("sparse_mlp": dense attention + hard-only MLP;
        "sparse_attn": hard-only queries and MLP, easy K/V frozen)
    5. easy tokens  -> draft eps        (identity easy handling; the sparse
       target never produces an output for easy tokens)
    6. hard tokens  -> sparse-target eps
    7. DDIM step

Selectors (score = priority for TARGET, i.e. estimated hardness):
    oracle    : true eps disagreement d_i (needs a dense target pass ->
                diagnostic only; the dense pass is NOT charged to the sparse
                system's FLOPs, but wall-clock is reported honestly per part)
    router    : learned draft-only router (--router ckpt)
    random    : random priorities
    frequency : wavelet HF energy of z_t per token (draft-free)
    norm      : draft-eps token norm (draft-only; NOTE deviation from the
                paper's token-norm which used the TARGET eps — a sparse system
                cannot see the target before selection)
    dense     : no sparsity — full target (reference; ignores r/m)
    draft     : draft only (ignores r/m)
    mix       : the paper's K=1 OUTPUT mixing with dense target (reference)

Per-step CUDA-event latency breakdown: draft / select / prefix / suffix / mix /
scheduler. Analytic MAC counts via models.dit_sparse.dit_model_flops.

Optionally dumps PNG samples per method for FID (clean-fid, same layout as
dit_token_fid.py).

Usage (oracle sparse-MLP sweep point, ImageNet-64):
    python dit_sparse_sampler.py \
        --target /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/target.pt --target_model DiT-S \
        --draft  /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/draft_nano.pt --draft_model DiT-Nano \
        --img_size 64 --patch 4 --num_classes 1000 \
        --selector oracle --suffix_mode sparse_mlp --hard_ratio 0.3 --split 0.5 \
        --n_samples 64 --out_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/sparse/oracle_mlp_r0.3_m0.5
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit import count_params
from models.dit_sparse import (dit_forward_tokens, sparse_target_eps,
                               sparse_target_eps_cached,
                               dit_forward_dense_with_cache,
                               dit_model_flops, topk_index)
from models.token_router import token_scalar_feats, build_router_from_ckpt
from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit, save_grid


class Timer:
    """CUDA-event (or wall) timer accumulating named segments in ms."""
    def __init__(self, dev):
        self.cuda = dev.type == "cuda"
        self.acc = {}

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def start(self):
        self._sync(); self.t0 = time.perf_counter()

    def stop(self, name):
        self._sync()
        self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - self.t0) * 1e3


def token_grid(z, p):
    return z.shape[-2] // p, z.shape[-1] // p


@torch.no_grad()
def select_hard(selector, args, ctx):
    """Return hard-token indices [B,k] (sorted) given per-step context."""
    z, tt, p = ctx["z"], ctx["tt"], args.patch
    h, w = token_grid(z, p)
    B, N = z.shape[0], h * w
    if selector == "oracle":
        d = F.avg_pool2d((ctx["eps_d"] - ctx["eps_t_dense"]).pow(2).mean(1, keepdim=True),
                         p, stride=p).flatten(1)
        score = d
    elif selector == "router":
        score = ctx["router"](ctx["h_d"].float(), ctx["scal"].float(), tt)
    elif selector == "anchor":
        score = ctx["anchor_score"]        # d_i measured at the last anchor
    elif selector == "delta":
        score = ctx["delta_score"]         # anchor-to-anchor target-eps change
    elif selector == "random":
        score = torch.rand(B, N, device=z.device)
    elif selector == "frequency":
        from models.wavelet import lwd_wavelet_saliency
        score = lwd_wavelet_saliency(z, ctx["dwt"], target_size=(h, w)).flatten(1)
    elif selector == "norm":
        score = F.avg_pool2d(ctx["eps_d"].pow(2).mean(1, keepdim=True),
                             p, stride=p).flatten(1)
    else:
        raise ValueError(selector)
    return torch.sort(topk_index(score, args.hard_ratio), dim=1).values


@torch.no_grad()
def sample_sparse(target, draft, sch, ts, y, z, args, dev, router=None):
    """One batch of samples with the requested method. Returns (x, stats)."""
    sel, mode = args.selector, args.suffix_mode
    p = args.patch
    L = len(target.blocks)
    m = max(0, min(L - 1, int(round(args.split * L))))
    tm = Timer(dev)
    dwt = None
    if sel == "frequency":
        from models.wavelet import DWT2D
        dwt = DWT2D("haar").to(dev)

    n_warm, n_anchor = 0, 0
    cnt = dict(target_dense=0, target_sparse=0, draft=0, oracle_diag=0)
    cache, since_anchor, anchor_score = None, 10**9, None
    eps_cache, prev_anchor_eps, delta_score = None, None, None
    reuse = (args.easy_source == "target_cache")
    for i, t in enumerate(ts):
        t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
        tt = torch.full((y.shape[0],), int(t), device=dev, dtype=torch.long)
        # phase schedule: dense-target warm-up while t/T > dense_until
        # (global layout formation; mirrors FreqSpec-Inpaint Phase 1)
        warm = (sel not in ("dense", "draft", "mix")
                and (int(t) / sch.num_train_timesteps) > args.dense_until)
        if warm:
            n_warm += 1
            tm.start()
            cnt["target_dense"] += 1
            if mode == "cache_attn":
                eps, cache = dit_forward_dense_with_cache(target, z, tt, y, m)
                since_anchor = 0
                if reuse:
                    if eps_cache is not None:
                        delta_score = F.avg_pool2d(
                            (eps - eps_cache).pow(2).mean(1, keepdim=True),
                            p, stride=p).flatten(1)
                    eps_cache = eps
            else:
                eps = target(z, tt, y)
            tm.stop("target_dense")
            tm.start(); z, _ = sch.ddim_step(z, eps, int(t), t_prev, eta=0.0)
            tm.stop("scheduler")
            continue

        # cache-mode anchor step: dense target + cache refresh, no draft
        if (sel not in ("dense", "draft", "mix") and mode == "cache_attn"
                and (cache is None or since_anchor >= args.cache_period - 1
                     or (sel == "anchor" and anchor_score is None)
                     or (sel == "delta" and delta_score is None))):
            n_anchor += 1
            cnt["target_dense"] += 1
            tm.start()
            eps, cache = dit_forward_dense_with_cache(target, z, tt, y, m)
            since_anchor = 0
            tm.stop("target_anchor")
            if reuse or sel == "delta":
                if eps_cache is not None:  # hardness = anchor-to-anchor change
                    delta_score = F.avg_pool2d(
                        (eps - eps_cache).pow(2).mean(1, keepdim=True),
                        p, stride=p).flatten(1)
                eps_cache = eps
            if sel == "anchor":            # measure exact d_i while both are cheap
                tm.start()
                cnt["draft"] += 1
                eps_da = draft(z, tt, y)
                anchor_score = F.avg_pool2d(
                    (eps_da - eps).pow(2).mean(1, keepdim=True),
                    p, stride=p).flatten(1)
                tm.stop("draft")
            tm.start(); z, _ = sch.ddim_step(z, eps, int(t), t_prev, eta=0.0)
            tm.stop("scheduler")
            continue

        if sel == "dense":
            cnt["target_dense"] += 1
            tm.start(); eps = target(z, tt, y); tm.stop("target_dense")
        elif sel == "draft":
            cnt["draft"] += 1
            tm.start(); eps = draft(z, tt, y); tm.stop("draft")
        elif sel == "mix":  # paper's K=1 output mixing (dense both)
            cnt["target_dense"] += 1; cnt["draft"] += 1
            tm.start(); eps_t = target(z, tt, y); tm.stop("target_dense")
            tm.start(); eps_d = draft(z, tt, y); tm.stop("draft")
            d = F.avg_pool2d((eps_d - eps_t).pow(2).mean(1, keepdim=True),
                             p, stride=p).flatten(1)
            idx = torch.sort(topk_index(d, args.hard_ratio), dim=1).values
            hard = torch.zeros_like(d, dtype=torch.bool).scatter_(1, idx, True)
            hf = hard.view(-1, 1, *token_grid(z, p)).float()
            hf = hf.repeat_interleave(p, 2).repeat_interleave(p, 3)
            eps = hf * eps_t + (1 - hf) * eps_d
        else:
            ctx = dict(z=z, tt=tt, router=router, dwt=dwt,
                       anchor_score=anchor_score, delta_score=delta_score)
            need_draft = (sel in ("router", "norm")
                          or (not reuse))          # draft supplies easy outputs
            eps_d = None
            if need_draft:
                tm.start()
                cnt["draft"] += 1
                if sel == "router":
                    eps_d, h_d = dit_forward_tokens(draft, z, tt, y)
                    ctx["h_d"] = h_d
                    ctx["scal"] = token_scalar_feats(eps_d, p, token_grid(z, p))
                else:
                    eps_d = draft(z, tt, y)
                ctx["eps_d"] = eps_d
                tm.stop("draft")
            if sel == "oracle":
                tm.start(); cnt["oracle_diag"] += 1
                ctx["eps_t_dense"] = target(z, tt, y)
                if reuse:                     # oracle risk vs the reused eps
                    ctx["eps_d"] = eps_cache
                tm.stop("oracle_dense_target")  # upper-bound-only cost
            tm.start(); idx = select_hard(sel, args, ctx); tm.stop("select")
            canvas = eps_cache if reuse else eps_d
            tm.start()
            cnt["target_sparse"] += 1
            if mode == "cache_attn":
                eps = sparse_target_eps_cached(target, z, tt, y, idx, m,
                                               canvas, cache)
                since_anchor += 1
            else:
                eps = sparse_target_eps(target, z, tt, y, idx, m, mode,
                                        canvas, args.refresh_every)
            tm.stop("target_sparse")
        tm.start(); z, _ = sch.ddim_step(z, eps, int(t), t_prev, eta=0.0)
        tm.stop("scheduler")
    return z, tm.acc, n_warm, n_anchor, cnt


def flops_summary(target, draft, args, steps, cnt):
    """Exact MAC accounting from counted forward passes. Deployable total
    excludes oracle diagnostic passes; they are reported separately so table
    captions can state the exclusion (oracle = selection upper bound)."""
    L = len(target.blocks)
    m = max(0, min(L - 1, int(round(args.split * L))))
    k = max(1, int(round(args.hard_ratio * target.num_tokens)))
    f_dense = dit_model_flops(target, "dense")
    f_draft = dit_model_flops(draft, "dense")
    smode = ("sparse_attn" if args.suffix_mode == "cache_attn"
             else args.suffix_mode)        # cached suffix = sparse_attn ops
    f_sparse = dit_model_flops(target, smode, m=m, k=k,
                               refresh_every=args.refresh_every)
    total = (cnt["target_dense"] * f_dense + cnt["target_sparse"] * f_sparse
             + cnt["draft"] * f_draft)
    return dict(target_dense_gmac=round(f_dense / 1e9, 4),
                draft_gmac=round(f_draft / 1e9, 4),
                per_step_gmac=round(total / steps / 1e9, 4),
                per_step_vs_dense=round(total / (f_dense * steps), 4),
                total_gmac=round(total / 1e9, 3),
                oracle_diag_gmac=round(cnt["oracle_diag"] * f_dense / 1e9, 3),
                calls=dict(cnt),
                split_m=m, hard_k=k,
                refresh_every=args.refresh_every)


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    router = build_router_from_ckpt(args.router, dev) if args.router else None
    if args.selector == "router" and router is None:
        raise ValueError("--selector router requires --router <ckpt>")
    if (args.easy_source == "target_cache" or args.selector == "delta") \
            and args.suffix_mode != "cache_attn" \
            and args.selector not in ("dense", "draft", "mix"):
        raise ValueError("--easy_source target_cache / --selector delta "
                         "require --suffix_mode cache_attn")
    print(f"[sparse] target {args.target_model} {count_params(target)/1e6:.1f}M | "
          f"draft {args.draft_model} {count_params(draft)/1e6:.1f}M | "
          f"selector={args.selector} suffix={args.suffix_mode} "
          f"r={args.hard_ratio} split={args.split}")

    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    os.makedirs(args.out_dir, exist_ok=True)
    dump_dir = os.path.join(args.out_dir, "samples") if args.dump_samples else None
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    torch.manual_seed(args.seed)  # random selector uses global RNG
    n_done, idx_img = 0, 0
    timing_total = {}
    xs = []
    t_wall0 = time.perf_counter()
    while n_done < args.n_samples:
        b = min(args.batch, args.n_samples - n_done)
        y = torch.randint(0, args.num_classes, (b,), device=dev, generator=g)
        z = torch.randn(b, 3, args.img_size, args.img_size, device=dev, generator=g)
        x, tacc, n_warm, n_anchor, cnt = sample_sparse(target, draft, sch, ts, y, z, args, dev, router)
        for k, v in tacc.items():
            timing_total[k] = timing_total.get(k, 0.0) + v
        if dump_dir:
            from torchvision.utils import save_image
            xp = (x.clamp(-1, 1) + 1) / 2
            for j in range(b):
                save_image(xp[j], os.path.join(dump_dir, f"{idx_img:06d}.png"))
                idx_img += 1
        elif len(xs) * args.batch < 64:
            xs.append(x.cpu())
        n_done += b
        print(f"[sparse] {n_done}/{args.n_samples}")
    wall = time.perf_counter() - t_wall0

    if xs:
        save_grid(torch.cat(xs)[:64], os.path.join(args.out_dir, "grid.png"))

    fl = flops_summary(target, draft, args, args.steps, cnt)
    per_img_ms = {k: round(v / args.n_samples, 2) for k, v in timing_total.items()}
    summary = dict(selector=args.selector, suffix_mode=args.suffix_mode,
                   hard_ratio=args.hard_ratio, split=args.split,
                   refresh_every=args.refresh_every, dense_until=args.dense_until,
                   cache_period=args.cache_period,
                   easy_source=args.easy_source,
                   steps=args.steps, n_samples=args.n_samples,
                   flops=fl, per_image_ms=per_img_ms,
                   wall_s=round(wall, 2),
                   wall_per_image_s=round(wall / args.n_samples, 4))
    fid = None
    if args.ref_dir and dump_dir:
        from cleanfid import fid as cfid
        fid = float(cfid.compute_fid(dump_dir, args.ref_dir, mode="clean",
                                     num_workers=args.fid_workers))
        summary["fid"] = round(fid, 3)
        print(f"[sparse] FID = {fid:.3f}")
    with open(os.path.join(args.out_dir, "sparse_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sparse] per-step FLOPs vs dense target: {fl['per_step_vs_dense']:.3f} "
          f"(split m={fl['split_m']}, k={fl['hard_k']})")
    print(f"[sparse] per-image ms: {per_img_ms}")
    print(f"[sparse] wrote {os.path.join(args.out_dir, 'sparse_summary.json')}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--router", type=str, default="",
                    help="router checkpoint (required for --selector router)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--selector", type=str, default="oracle",
                    choices=["dense", "draft", "mix", "oracle", "router",
                             "random", "frequency", "norm", "anchor",
                             "delta"])
    ap.add_argument("--suffix_mode", type=str, default="sparse_mlp",
                    choices=["dense", "sparse_mlp", "sparse_attn", "cache_attn"])
    ap.add_argument("--hard_ratio", type=float, default=0.3,
                    help="fraction of tokens executed by the target suffix")
    ap.add_argument("--split", type=float, default=0.5,
                    help="prefix depth fraction m/L (dense prefix)")
    ap.add_argument("--refresh_every", type=int, default=0,
                    help="run every s-th suffix block densely (0 = off); "
                         "bounds easy-token staleness")
    ap.add_argument("--easy_source", type=str, default="draft",
                    choices=["draft", "target_cache"],
                    help="easy-token eps source: the draft prediction, or the "
                         "target's own eps from the last anchor (draft-free "
                         "token-wise step allocation; cache_attn only)")
    ap.add_argument("--cache_period", type=int, default=5,
                    help="cache_attn: dense anchor every this many steps; "
                         "easy-token suffix context is reused from the last "
                         "anchor (depth-correct, time-stale)")
    ap.add_argument("--dense_until", type=float, default=1.0,
                    help="run the dense target while t/T is above this value "
                         "(phase warm-up; 1.0 = off, 0.7 = FreqSpec-style)")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump_samples", action="store_true",
                    help="write PNGs (needed for FID)")
    ap.add_argument("--ref_dir", type=str, default="",
                    help="reference image folder; computes clean-fid if set "
                         "(requires --dump_samples)")
    ap.add_argument("--fid_workers", type=int, default=0,
                    help="keep 0 on Python 3.14 (forkserver pickling issue)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
