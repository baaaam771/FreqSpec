#!/usr/bin/env python
"""
dit_inpaint_sparse.py — token-selective correction for DiT inpainting
(Stages 2--4), on top of the anchored-reuse machinery.

One pipeline for EVERY method (dense, r=0 reuse, all selectors), so the
known-region reinjection, mask generation, scheduler, and seeds are
bit-identical across comparisons -- the fairness precondition for reusing
the Stage-1 dense baselines (rerun dense here only if Stage 1 used a
different reinjection).

Inpainting loop (RePaint-style reinjection, every step):
    z_{t'} <- M .* z_{t'}^{gen} + (1-M) .* q_sample(x_known, t')
with M=1 inside the hole. Final composite: M.*x_gen + (1-M).*x_known.

Feedback items implemented:
  1. frequency selector computed on the ANCHOR's predicted clean latent
     x0_a = (z_a - sqrt(1-abar_a) eps_a) / sqrt(abar_a), not on z_t.
  2. --budget_scope {global, mask}: global ranks all N tokens; mask
     restricts selection to hole-overlapping tokens only (known tokens are
     overwritten by reinjection and must never consume budget).
  3. exact mask-only budget: per-image k_i = round(r * |mask tokens_i|),
     realized as a rectangular gather via top-1 padding; FLOPs use true k_i.
  4. combo score: per-image RANK-normalized components before weighting
     (score scales are incomparable otherwise):
       combo = w_d rank(delta) + w_f rank(freq) + w_b rank(boundary).
  5. selectors {boundary, frequency, delta, combo, random, rotate, oracle}
     for the ablation; oracle = ||eps_cache - eps_dense_now|| (reuse-regime
     oracle; diagnostic pass excluded from deployable TOTAL).

CFG: --cfg 1.0 disables (default). With cfg>1 every model evaluation is
internally batch-doubled (cond + null class) and counters are doubled.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit_sparse import (dit_forward_dense_with_cache,
                               sparse_target_eps_cached, dit_model_flops)
from models.token_router import rank_normalize
from models.wavelet import DWT2D, lwd_wavelet_saliency
from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit
from inpaint_masks import make_mask_batch, token_mask

SELECTORS = ["dense", "r0", "delta", "frequency", "boundary", "combo",
             "random", "rotate", "oracle"]


# --------------------------------------------------------------- data utils
def load_images(data_dir, list_file, indices, img_size, device):
    with open(list_file) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    xs = []
    for i in indices:
        img = Image.open(os.path.join(data_dir, names[i])).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BICUBIC)
        xs.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1))
    x = torch.stack(xs).float().to(device) / 127.5 - 1.0
    return x, [names[i] for i in indices]


def load_labels(labels_file, indices, device, null_class):
    if not labels_file:
        return torch.full((len(indices),), null_class, device=device,
                          dtype=torch.long)
    with open(labels_file) as f:
        labs = [int(ln.strip()) for ln in f if ln.strip()]
    return torch.tensor([labs[i] for i in indices], device=device)


# ------------------------------------------------------------- mask budget
def masked_topk(score, sel_mask, ratio):
    """Exact per-image budget inside sel_mask. score,[B,N]; sel_mask bool.
    k_i = round(ratio * |sel_mask_i|); rectangular [B,kmax] via top-1
    padding (duplicate indices are idempotent under scatter). Returns
    (sorted idx [B,kmax], true k_i [B])."""
    neg = torch.finfo(score.dtype).min
    s = score.masked_fill(~sel_mask, neg)
    n_sel = sel_mask.sum(1).clamp(min=1)
    k_i = (ratio * n_sel).round().clamp(min=1).long()
    kmax = int(k_i.max())
    idx = s.topk(kmax, dim=1).indices
    pad = torch.arange(kmax, device=score.device)[None] >= k_i[:, None]
    idx = torch.where(pad, idx[:, :1].expand_as(idx), idx)
    return torch.sort(idx, dim=1).values, k_i


# ------------------------------------------------------------- CFG wrapper
class Guided:
    """Batch-doubled classifier-free guidance around the cache machinery."""
    def __init__(self, scale, null_class):
        self.s, self.null = scale, null_class
        self.on = scale is not None and scale > 1.0

    def pack(self, z, y):
        if not self.on:
            return z, y
        return torch.cat([z, z]), torch.cat(
            [y, torch.full_like(y, self.null)])

    def unpack(self, eps):
        if not self.on:
            return eps
        c, u = eps.chunk(2)
        return u + self.s * (c - u)


# --------------------------------------------------------------- selectors
def compute_scores(sel, ctx, args):
    """Return [B,N] priority score for the requested selector."""
    B, N = ctx["B"], ctx["N"]
    dev = ctx["dev"]
    if sel == "random":
        return torch.rand(B, N, device=dev)
    if sel == "rotate":
        n_grp = max(1, int(round(1.0 / max(args.hard_ratio, 1e-6))))
        grp = ctx["sparse_step_idx"] % n_grp
        base = (torch.arange(N, device=dev) % n_grp == grp).float()
        return base.unsqueeze(0).expand(B, -1) \
            + 1e-3 * torch.rand(B, N, device=dev)
    if sel == "delta":
        return ctx["delta_score"]
    if sel == "frequency":
        return ctx["freq_score"]           # HF of anchor x0 (item 1)
    if sel == "boundary":
        return ctx["band_tok"].float() \
            + 1e-3 * torch.rand(B, N, device=dev)
    if sel == "combo":                     # item 4: rank-normalized mix
        comps, ws = [], []
        if ctx["delta_score"] is not None and args.w_delta > 0:
            comps.append(rank_normalize(ctx["delta_score"]))
            ws.append(args.w_delta)
        if args.w_freq > 0:
            comps.append(rank_normalize(ctx["freq_score"]))
            ws.append(args.w_freq)
        if args.w_boundary > 0:
            comps.append(rank_normalize(
                ctx["band_tok"].float()
                + 1e-4 * torch.rand(B, N, device=dev)))
            ws.append(args.w_boundary)
        return sum(w * c for w, c in zip(ws, comps)) / max(sum(ws), 1e-9)
    if sel == "oracle":
        return ctx["oracle_score"]
    raise ValueError(sel)


# ------------------------------------------------------------------- main
@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4,
                       beta_end=0.02, beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = target                                   # draft-free line
    dwt = DWT2D("haar").to(dev)
    p = args.patch
    hw = (args.img_size // p, args.img_size // p)
    N = hw[0] * hw[1]
    guide = Guided(args.cfg, args.num_classes)       # null = class index C
    cfg_mult = 2 if guide.on else 1
    m_split = max(0, min(len(target.blocks) - 1,
                         int(round(args.split * len(target.blocks)))))

    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "masks"), exist_ok=True)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    torch.manual_seed(args.seed)

    cnt = dict(target_dense=0, target_sparse=0, oracle_diag=0)
    true_k_sum, true_k_n = 0.0, 0
    names_all = []
    t_wall0 = time.perf_counter()
    n_done = 0
    while n_done < args.n_samples:
        b = min(args.batch, args.n_samples - n_done)
        idxs = list(range(n_done, n_done + b))
        x0, names = load_images(args.data_dir, args.list_file, idxs,
                                args.img_size, dev)
        y = load_labels(args.labels_file, idxs, dev, args.num_classes)
        M = make_mask_batch(idxs, args.img_size, args.img_size, dev,
                            kind=args.mask_kind, seed=args.mask_seed,
                            amin=args.mask_amin, amax=args.mask_amax)
        mask_tok, band_tok = token_mask(M, p)        # [B,N] bool each
        sel_mask = mask_tok if args.budget_scope == "mask" \
            else torch.ones_like(mask_tok)           # item 2

        z = torch.randn(b, 3, args.img_size, args.img_size, device=dev,
                        generator=g)
        # initial reinjection at t=T
        eps0 = torch.randn(z.shape, device=dev, generator=g)
        tT = torch.full((b,), int(ts[0]), device=dev, dtype=torch.long)
        z = M * z + (1 - M) * sch.q_sample(x0, eps0, tT)

        cache, since_anchor, sparse_i = None, 10**9, 0
        eps_cache, delta_score, freq_score = None, None, None
        for i, t in enumerate(ts):
            t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
            tt = torch.full((b,), int(t), device=dev, dtype=torch.long)
            need_anchor = (args.selector != "dense"
                           and (cache is None
                                or since_anchor >= args.cache_period - 1
                                or (args.selector == "delta"
                                    and args.hard_ratio > 0
                                    and delta_score is None)))
            if args.selector == "dense" or need_anchor:
                z2, y2 = guide.pack(z, y)
                t2 = tt.repeat(cfg_mult)
                eps2, cache = dit_forward_dense_with_cache(
                    target, z2, t2, y2, m_split)
                eps = guide.unpack(eps2)
                cnt["target_dense"] += cfg_mult
                since_anchor = 0
                if args.selector != "dense":
                    if eps_cache is not None:
                        delta_score = F.avg_pool2d(
                            (eps - eps_cache).pow(2).mean(1, keepdim=True),
                            p, stride=p).flatten(1)
                    eps_cache = eps
                    # item 1: frequency on the anchor's predicted x0
                    sa = sch.sqrt_alphas_cumprod[tt].view(-1, 1, 1, 1)
                    som = sch.sqrt_one_minus_alphas_cumprod[tt].view(
                        -1, 1, 1, 1)
                    x0_a = ((z - som * eps) / sa).clamp(-1.5, 1.5)
                    freq_score = lwd_wavelet_saliency(
                        x0_a, dwt, target_size=hw).flatten(1)
            else:
                if args.hard_ratio <= 0 or args.selector == "r0":
                    eps = eps_cache                   # pure anchored reuse
                else:
                    ctx = dict(B=b, N=N, dev=dev, delta_score=delta_score,
                               freq_score=freq_score, band_tok=band_tok,
                               sparse_step_idx=sparse_i,
                               oracle_score=None)
                    if args.selector == "oracle":
                        z2, y2 = guide.pack(z, y)
                        ed = guide.unpack(target(z2, tt.repeat(cfg_mult), y2))
                        cnt["oracle_diag"] += cfg_mult
                        ctx["oracle_score"] = F.avg_pool2d(
                            (eps_cache - ed).pow(2).mean(1, keepdim=True),
                            p, stride=p).flatten(1)
                    score = compute_scores(args.selector, ctx, args)
                    idx, k_i = masked_topk(score, sel_mask,
                                           args.hard_ratio)    # item 3
                    true_k_sum += float(k_i.float().mean())
                    true_k_n += 1
                    z2, y2 = guide.pack(z, y)
                    idx2 = idx.repeat(cfg_mult, 1)
                    canvas2 = eps_cache.repeat(cfg_mult, 1, 1, 1) \
                        if guide.on else eps_cache
                    eps2 = sparse_target_eps_cached(
                        target, z2, tt.repeat(cfg_mult), y2, idx2,
                        m_split, canvas2, cache)
                    eps = guide.unpack(eps2)
                    cnt["target_sparse"] += cfg_mult
                since_anchor += 1
                sparse_i += 1
            z, _ = sch.ddim_step(z, eps, int(t), t_prev, eta=0.0)
            # per-step known-region reinjection (identical for ALL methods)
            if t_prev >= 0:
                epsk = torch.randn(z.shape, device=dev, generator=g)
                tp = torch.full((b,), t_prev, device=dev, dtype=torch.long)
                z = M * z + (1 - M) * sch.q_sample(x0, epsk, tp)
            else:
                z = M * z + (1 - M) * x0

        xout = z.clamp(-1, 1)
        for j in range(b):
            im = ((xout[j].cpu().permute(1, 2, 0).numpy() + 1) * 127.5
                  ).astype(np.uint8)
            Image.fromarray(im).save(
                os.path.join(args.out_dir, "samples", f"{n_done+j:06d}.png"))
            mk = (M[j, 0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mk).save(
                os.path.join(args.out_dir, "masks", f"{n_done+j:06d}.png"))
        names_all.extend(names)
        n_done += b
        print(f"[inp-sparse] {args.selector} {n_done}/{args.n_samples}")

    wall = time.perf_counter() - t_wall0
    f_dense = dit_model_flops(target, "dense")
    k_mean = true_k_sum / max(true_k_n, 1)
    f_sparse = dit_model_flops(target, "sparse_attn", m=m_split,
                               k=max(int(round(k_mean)), 1))
    total = cnt["target_dense"] * f_dense + cnt["target_sparse"] * f_sparse
    denom = f_dense * args.steps * (args.n_samples / args.batch) * cfg_mult
    summary = dict(selector=args.selector, budget_scope=args.budget_scope,
                   hard_ratio=args.hard_ratio, steps=args.steps,
                   cache_period=args.cache_period, split=args.split,
                   cfg=args.cfg, mask_kind=args.mask_kind,
                   mask_seed=args.mask_seed, n_samples=args.n_samples,
                   mean_true_k=round(k_mean, 2), calls=cnt,
                   total_vs_dense_same_steps=round(total / denom, 4),
                   wall_s=round(wall, 1),
                   weights=dict(delta=args.w_delta, freq=args.w_freq,
                                boundary=args.w_boundary))
    with open(os.path.join(args.out_dir, "inpaint_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "gt_names.txt"), "w") as f:
        f.write("\n".join(names_all))
    print(f"[inp-sparse] TOTAL vs dense@{args.steps}steps: "
          f"{summary['total_vs_dense_same_steps']} | mean k {k_mean:.1f} "
          f"| wrote {args.out_dir}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft", type=str, default="")   # unused (draft-free)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--list_file", type=str, required=True,
                    help="validation image list (Stage-1 invariant)")
    ap.add_argument("--labels_file", type=str, default="",
                    help="one class id per line; empty = null class")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--selector", type=str, default="delta",
                    choices=SELECTORS)
    ap.add_argument("--budget_scope", type=str, default="mask",
                    choices=["global", "mask"])
    ap.add_argument("--hard_ratio", type=float, default=0.3,
                    help="fraction of the SCOPE (mask tokens if scope=mask)")
    ap.add_argument("--w_delta", type=float, default=1.0)
    ap.add_argument("--w_freq", type=float, default=1.0)
    ap.add_argument("--w_boundary", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cache_period", type=int, default=2)
    ap.add_argument("--split", type=float, default=0.0)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--mask_kind", type=str, default="mixed",
                    choices=["box", "freeform", "mixed"])
    ap.add_argument("--mask_seed", type=int, default=0)
    ap.add_argument("--mask_amin", type=float, default=0.05)
    ap.add_argument("--mask_amax", type=float, default=0.35)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
