#!/usr/bin/env python
"""
dit_inpaint_sampler.py — DACE (depth-aligned cached execution) applied to
DiT INPAINTING on ImageNet-64: the deployment the DACE paper predicts to be
favorable ("masked regions change fast, context barely moves").

Pipeline per batch of validation images:
    x0, y from ImageFolder; mask sampled deterministically (mask_seed)
    z_T ~ N(0,1) (run_seed);  known-region re-injection at every step:
        z <- mask * z + (1-mask) * (sqrt(abar_t) x0 + sqrt(1-abar_t) eps0)

Execution modes (--mode):
    dense   : dense target at --steps S             (step-reduction baseline)
    dace    : anchor every c steps = dense pass with suffix-input cache;
              between anchors = dense prefix (blocks 0..m-1) + cached sparse
              suffix on hard tokens only. Easy-token eps source (--easy):
                anchor : the target's own anchor prediction (draft-free,
                         target-eps reuse — the DACE Sec. 4.5 regime)
                draft  : draft prediction (draft runs dense at sparse steps)
    mix     : both models dense, output mixing (verifier ceiling reference)
    draft   : draft only

Selectors (--selector), score = priority for target refresh:
    mask     : dilated hole-token interior fraction (+boundary tiebreak);
               the mask is the training-free spatial router inpainting
               provides for free (Stage-2 question of the plan)
    boundary : token boundary band only
    freq     : wavelet HF energy of z_t per token (draft-free)
    delta    : anchor-to-anchor change of the TARGET's own prediction
               ||eps_a - eps_{a-}||^2 per token (draft-free; falls back to
               mask score before the second anchor)
    combo    : a*Mask + b*Boundary + g*rank(Freq) + d*rank(Delta)
    anchor   : draft-target disagreement measured at the last anchor,
               carried forward (one draft pass per anchor)
    random   : random priorities
    oracle   : true per-token eps disagreement vs a dense target pass at the
               CURRENT step (diagnostic upper bound; the dense pass is not
               charged to MACs, wall-clock reported separately)

Hard budget: --hard_ratio r fixes k = ceil(r*N) (rectangular gather).
    --hard_ratio 0  = AUTO: k = max over batch of |dilate(mask tokens)| so
    every hole token is refreshed every sparse step (pure mask-only routing).
    --restrict_to_mask adds a large bias so the budget is spent inside
    dilate(mask) first (outside tokens only if budget exceeds the hole).
    --block b rounds the hard set up to bxb token blocks (structured
    sparsity; contiguous inpainting masks make this cheap).

Metrics (per method, written to metrics.json):
    mask-PSNR / mask-LPIPS vs ground truth
    mask-MSE_t / mask-LPIPS_t vs the dense-50 reference trajectory
        (same run_seed / mask_seed / eps0 — cached in --ref_dir)
    MACs relative to 50-step dense target; NFE breakdown; CUDA wall-clock
    PNG dump of completed images for FID (clean-fid vs val set)

Usage (server):
    python dit_inpaint_sampler.py \
        --target /mnt/HDD_12TB/bam_ki/ckpt_dit_inp/target.pt --target_model DiT-S-Inp \
        --draft  /mnt/HDD_12TB/bam_ki/ckpt_dit_inp/draft_nano.pt --draft_model DiT-Nano-Inp \
        --data_root /mnt/HDD_12TB/bam_ki/imagenet64/val \
        --mode dace --selector mask --easy anchor --steps 30 --cache_period 2 \
        --split 0 --hard_ratio 0.3 --n_samples 200 \
        --out_dir /mnt/HDD_12TB/bam_ki/results/dit_inp/dace_mask_r0.3
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit_inpaint import DiTInpaint, load_dit_inpaint
from models.dit_sparse import (dit_forward_dense_with_cache,
                               sparse_target_eps_cached, dit_model_flops,
                               topk_index)
from training.scheduler import DDPMSchedule
from utils.inpaint_masks import (sample_masks, mask_to_tokens, dilate_tokens,
                                 boundary_band, combo_score, rank_normalize,
                                 block_round_indices, known_latent)

try:
    from models.wavelet import DWT2D
    _HAS_DWT = True
except Exception:
    _HAS_DWT = False

try:
    import lpips as lpips_lib
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


# ------------------------------------------------------------------ utilities
class Timer:
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


def get_val_loader(args):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize(args.img_size),
                             transforms.CenterCrop(args.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    g = torch.Generator().manual_seed(args.data_seed)
    idx = torch.randperm(len(ds), generator=g)[:args.n_samples].tolist()
    sub = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(sub, batch_size=args.batch,
                                       shuffle=False, num_workers=args.workers,
                                       drop_last=False)


def freq_token_score(z, h, w, dwt):
    """Wavelet HF energy per token [B,N] (fallback: Laplacian energy)."""
    if dwt is not None:
        _, lh, hl, hh = dwt(z)
        e = (lh ** 2 + hl ** 2 + hh ** 2).mean(1, keepdim=True)
    else:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=z.device).view(1, 1, 3, 3).repeat(z.shape[1], 1, 1, 1)
        e = F.conv2d(z, k, padding=1, groups=z.shape[1]).pow(2).mean(1, keepdim=True)
    return F.adaptive_avg_pool2d(e, (h, w)).flatten(1)


def per_token_sq(diff, p):
    """[B,C,H,W] -> per-token mean squared value [B,N]."""
    return F.avg_pool2d(diff.pow(2).mean(1, keepdim=True), p, stride=p).flatten(1)


def masked_psnr(a, b, m):
    """a,b in [-1,1], m [B,1,H,W]; PSNR over masked pixels, per batch mean."""
    mse = ((a - b) ** 2 * m).flatten(1).sum(1) / (m.flatten(1).sum(1) * a.shape[1] + 1e-8)
    return (10 * torch.log10(4.0 / (mse + 1e-12))).mean().item()


# --------------------------------------------------------------- selector core
@torch.no_grad()
def selector_score(args, st):
    """Return hard-token priority score [B,N] for the CURRENT sparse step.
    st = dict of per-batch state."""
    sel, B, N = args.selector, st["B"], st["N"]
    h, w = st["hw"]
    dev = st["z"].device
    m_tok = st["mask_tok_dil"]                       # [B,N] dilated interior
    b_tok = st["bnd_tok"]                            # [B,N] boundary band

    if sel == "mask":
        score = m_tok + 0.5 * b_tok + 1e-4 * torch.rand(B, N, device=dev)
    elif sel == "boundary":
        score = b_tok + 1e-4 * torch.rand(B, N, device=dev)
    elif sel == "freq":
        score = freq_token_score(st["z"], h, w, st["dwt"])
    elif sel == "delta":
        score = st["delta_tok"] if st["delta_tok"] is not None \
            else m_tok + 0.5 * b_tok
    elif sel == "combo":
        f = freq_token_score(st["z"], h, w, st["dwt"])
        score = combo_score(M_tok=m_tok, B_tok=b_tok, F_tok=f,
                            D_tok=st["delta_tok"],
                            a=args.cw_mask, b=args.cw_bnd,
                            g=args.cw_freq, d=args.cw_delta)
    elif sel == "anchor":
        score = st["anchor_dis_tok"] if st["anchor_dis_tok"] is not None \
            else m_tok + 0.5 * b_tok
    elif sel == "random":
        score = torch.rand(B, N, device=dev)
    elif sel == "oracle":
        eps_t_dense = st["target"](st["x_in"], st["tt"], st["y"])
        st["oracle_extra_nfe"] += 1
        score = per_token_sq(st["eps_easy_now"] - eps_t_dense, args.patch)
    else:
        raise ValueError(f"unknown selector {sel}")

    if args.restrict_to_mask and sel not in ("mask", "boundary"):
        score = score + 1e6 * m_tok
    return score


# ------------------------------------------------------------------- sampling
@torch.no_grad()
def run_batch(args, target, draft, sch, ts, x0, y, mask, gen, dev, timer,
              dense_only_steps=None):
    """Sample one batch. dense_only_steps: if not None, ignore args.mode and
    run a plain dense trajectory with this step count (used for the dense-50
    reference). Returns (x_out [B,3,H,W], stats dict)."""
    B = x0.shape[0]
    p = args.patch
    h = w = args.img_size // p
    N = h * w

    x_masked = x0 * (1 - mask)
    z = torch.randn(B, 3, args.img_size, args.img_size, generator=gen,
                    device=dev)
    eps0 = torch.randn(B, 3, args.img_size, args.img_size, generator=gen,
                       device=dev)
    sac = sch.sqrt_alphas_cumprod
    s1m = sch.sqrt_one_minus_alphas_cumprod

    schedule = ts if dense_only_steps is None else \
        sch.get_ddim_schedule_exact(dense_only_steps)
    mode = "dense" if dense_only_steps is not None else args.mode

    # token-level mask machinery (fixed per batch)
    m_tok_frac = mask_to_tokens(mask, p)                       # [B,1,h,w]
    m_tok_bin = (m_tok_frac > args.tok_thresh).float()
    m_tok_dil = dilate_tokens(m_tok_bin, args.mask_dilate)
    bnd = boundary_band(m_tok_bin, args.boundary_k)
    dwt = DWT2D("haar").to(dev) if _HAS_DWT else None

    st = dict(B=B, N=N, hw=(h, w), target=target, y=y, dwt=dwt,
              mask_tok_dil=m_tok_dil.flatten(1), bnd_tok=bnd.flatten(1),
              delta_tok=None, anchor_dis_tok=None, oracle_extra_nfe=0)

    # hard-budget k
    if args.hard_ratio > 0:
        k = max(1, int(round(args.hard_ratio * N)))
    else:  # AUTO: cover every dilated hole token
        k = max(1, int(m_tok_dil.flatten(1).sum(1).max().item()))
    stats = dict(tgt_dense_nfe=0, tgt_sparse_nfe=0, drf_nfe=0, macs=0.0,
                 k=k, oracle_extra_nfe=0)
    macs_dense_t = dit_model_flops(target, "dense")
    macs_dense_d = dit_model_flops(draft, "dense") if draft is not None else 0

    # known-region injection at the start
    t0 = int(schedule[0].item())
    z = mask * z + (1 - mask) * known_latent(x0, eps0, sac, s1m, t0)

    cache, eps_anchor, eps_anchor_prev = None, None, None
    for i in range(len(schedule)):
        t = int(schedule[i].item())
        t_prev = int(schedule[i + 1].item()) if i + 1 < len(schedule) else -1
        tt = torch.full((B,), t, device=dev, dtype=torch.long)
        x_in = DiTInpaint.pack(z, x_masked, mask)
        st.update(z=z, x_in=x_in, tt=tt)

        is_anchor = (mode != "dace") or (i % args.cache_period == 0)

        if mode == "dense" or (mode == "dace" and is_anchor):
            timer.start()
            if mode == "dace":
                eps, cache = dit_forward_dense_with_cache(target, x_in, tt, y,
                                                          args.split_m)
                # delta selector bookkeeping
                if eps_anchor is not None:
                    st["delta_tok"] = per_token_sq(eps - eps_anchor, p)
                eps_anchor_prev, eps_anchor = eps_anchor, eps
                if args.selector == "anchor":
                    eps_d = draft(x_in, tt, y)
                    stats["drf_nfe"] += 1
                    stats["macs"] += macs_dense_d
                    st["anchor_dis_tok"] = per_token_sq(eps_d - eps, p)
            else:
                eps = target(x_in, tt, y)
            timer.stop("target_dense")
            stats["tgt_dense_nfe"] += 1
            stats["macs"] += macs_dense_t

        elif mode == "dace":                     # sparse step
            # easy-token eps source
            if args.easy == "anchor":
                eps_easy = eps_anchor
            elif args.easy == "draft":
                timer.start()
                eps_easy = draft(x_in, tt, y)
                timer.stop("draft")
                stats["drf_nfe"] += 1
                stats["macs"] += macs_dense_d
            else:
                raise ValueError(args.easy)
            st["eps_easy_now"] = eps_easy

            timer.start()
            score = selector_score(args, st)
            idx = topk_index(score, k / N)
            if args.block > 1:
                idx = block_round_indices(idx, h, w, args.block)
            else:
                idx = torch.sort(idx, dim=1).values
            timer.stop("select")

            timer.start()
            eps = sparse_target_eps_cached(target, x_in, tt, y, idx,
                                           args.split_m, eps_easy.clone(),
                                           cache)
            timer.stop("target_sparse")
            stats["tgt_sparse_nfe"] += 1
            stats["macs"] += dit_model_flops(target, "sparse_attn",
                                             m=args.split_m,
                                             k=idx.shape[1])

        elif mode == "mix":                      # output-mixing ceiling
            timer.start()
            eps_t = target(x_in, tt, y)
            eps_d = draft(x_in, tt, y)
            timer.stop("dense_both")
            stats["tgt_dense_nfe"] += 1; stats["drf_nfe"] += 1
            stats["macs"] += macs_dense_t + macs_dense_d
            st["eps_easy_now"] = eps_d
            dis = per_token_sq(eps_d - eps_t, p)
            idx = torch.sort(topk_index(dis, k / N), dim=1).values
            # scatter target tokens into draft canvas in token space
            from models.dit_sparse import patchify_img, scatter_tokens, gather_tokens
            cv = patchify_img(eps_d, p)
            tv = patchify_img(eps_t, p)
            cv = scatter_tokens(cv, idx, gather_tokens(tv, idx))
            eps = target.unpatchify(cv, (h, w))

        elif mode == "draft":
            timer.start()
            eps = draft(x_in, tt, y)
            timer.stop("draft")
            stats["drf_nfe"] += 1
            stats["macs"] += macs_dense_d
        else:
            raise ValueError(mode)

        timer.start()
        z, _ = sch.ddim_step(z, eps, t, t_prev)
        if not args.no_reinject:
            z = mask * z + (1 - mask) * known_latent(x0, eps0, sac, s1m,
                                                     t_prev)
        timer.stop("scheduler")

    stats["oracle_extra_nfe"] = st["oracle_extra_nfe"]
    x_out = mask * z + (1 - mask) * x0
    return x_out.clamp(-1, 1), stats


# ----------------------------------------------------------------------- main
@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png"); os.makedirs(png_dir, exist_ok=True)
    if args.ref_dir:
        os.makedirs(args.ref_dir, exist_ok=True)

    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4,
                       beta_end=0.02, beta_schedule="linear", device=dev)
    ts = sch.get_ddim_schedule_exact(args.steps)

    target = load_dit_inpaint(args.target, args.target_model, args.img_size,
                              args.patch, args.num_classes, dev)
    draft = None
    if args.draft:
        draft = load_dit_inpaint(args.draft, args.draft_model, args.img_size,
                                 args.patch, args.num_classes, dev)

    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev).eval() if _HAS_LPIPS else None

    loader = get_val_loader(args)
    timer = Timer(dev)
    agg = dict(n=0, mask_psnr=0.0, mask_lpips=0.0, mask_mse_t=0.0,
               mask_lpips_t=0.0, macs=0.0, tgt_dense_nfe=0.0,
               tgt_sparse_nfe=0.0, drf_nfe=0.0, k=0.0)
    macs_ref = 50 * dit_model_flops(target, "dense")

    from torchvision.utils import save_image
    img_id = 0
    for bi, (x0, y) in enumerate(loader):
        x0, y = x0.to(dev), y.to(dev)
        B = x0.shape[0]
        mask = sample_masks(B, args.img_size, args.img_size,
                            box_prob=args.box_prob,
                            seed=args.mask_seed + bi).to(dev)

        # --- dense-50 reference trajectory (cached on disk) ---
        ref_path = os.path.join(args.ref_dir or args.out_dir,
                                f"ref_b{bi}_s{args.run_seed}_m{args.mask_seed}"
                                f"_n{args.n_samples}_d{args.data_seed}.pt")
        if os.path.exists(ref_path):
            x_ref = torch.load(ref_path, map_location=dev)
        else:
            gen_r = torch.Generator(device=dev).manual_seed(args.run_seed + bi)
            x_ref, _ = run_batch(args, target, draft, sch, ts, x0, y, mask,
                                 gen_r, dev, Timer(dev), dense_only_steps=50)
            torch.save(x_ref.cpu(), ref_path)
            x_ref = x_ref.to(dev)

        # --- method trajectory (same seeds) ---
        gen = torch.Generator(device=dev).manual_seed(args.run_seed + bi)
        x_out, stats = run_batch(args, target, draft, sch, ts, x0, y, mask,
                                 gen, dev, timer)

        # --- metrics ---
        agg["mask_psnr"] += masked_psnr(x_out, x0, mask) * B
        agg["mask_mse_t"] += (((x_out - x_ref) ** 2 * mask).flatten(1).sum(1)
                              / (mask.flatten(1).sum(1) * 3 + 1e-8)).sum().item()
        if lpips_fn is not None:
            lp_gt = lpips_fn(x_out * mask + (-1) * (1 - mask),
                             x0 * mask + (-1) * (1 - mask))
            lp_t = lpips_fn(x_out * mask + (-1) * (1 - mask),
                            x_ref * mask + (-1) * (1 - mask))
            agg["mask_lpips"] += lp_gt.sum().item()
            agg["mask_lpips_t"] += lp_t.sum().item()
        for kk in ("macs", "tgt_dense_nfe", "tgt_sparse_nfe", "drf_nfe", "k"):
            agg[kk] += stats[kk]
        agg["n"] += B

        for j in range(B):
            save_image(x_out[j] * 0.5 + 0.5,
                       os.path.join(png_dir, f"{img_id:06d}.png"))
            img_id += 1
        if bi == 0 and args.save_debug_grid:
            dbg = torch.cat([x0[:8] * (1 - mask[:8]) - mask[:8],
                             x_ref[:8], x_out[:8], x0[:8]], 0)
            save_image(dbg * 0.5 + 0.5,
                       os.path.join(args.out_dir, "debug_grid.png"), nrow=8)
        print(f"[batch {bi}] done ({agg['n']}/{args.n_samples})")

    n = agg["n"]
    nb = math.ceil(n / args.batch)
    res = dict(
        config=vars(args),
        n=n,
        mask_psnr=agg["mask_psnr"] / n,
        mask_lpips=(agg["mask_lpips"] / n) if lpips_fn else None,
        mask_mse_t=agg["mask_mse_t"] / n,
        mask_lpips_t=(agg["mask_lpips_t"] / n) if lpips_fn else None,
        macs_per_image=agg["macs"] / nb,
        macs_vs_dense50=(agg["macs"] / nb) / macs_ref,
        tgt_dense_nfe=agg["tgt_dense_nfe"] / nb,
        tgt_sparse_nfe=agg["tgt_sparse_nfe"] / nb,
        drf_nfe=agg["drf_nfe"] / nb,
        hard_k=agg["k"] / nb,
        wallclock_ms=dict(timer.acc),
        lpips_available=_HAS_LPIPS, dwt_available=_HAS_DWT,
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({kk: res[kk] for kk in
                      ("mask_psnr", "mask_lpips", "mask_mse_t", "mask_lpips_t",
                       "macs_vs_dense50", "tgt_dense_nfe", "tgt_sparse_nfe",
                       "drf_nfe", "hard_k")}, indent=2))
    print(f"[done] {args.out_dir}  (FID: clean-fid vs val PNGs in {png_dir})")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--target_model", type=str, default="DiT-S-Inp")
    p.add_argument("--draft", type=str, default="")
    p.add_argument("--draft_model", type=str, default="DiT-Nano-Inp")
    p.add_argument("--data_root", type=str, required=True,
                   help="ImageFolder validation root")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ref_dir", type=str, default="",
                   help="cache dir for dense-50 reference outputs (shared "
                        "across sweep points to avoid recomputation)")
    # execution
    p.add_argument("--mode", type=str, default="dace",
                   choices=["dense", "dace", "mix", "draft"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cache_period", type=int, default=2,
                   help="anchor period c (DACE)")
    p.add_argument("--split_m", type=int, default=0,
                   help="dense-prefix depth m; 0 = whole network sparse "
                        "(best point in the DACE reuse regime)")
    p.add_argument("--easy", type=str, default="anchor",
                   choices=["anchor", "draft"],
                   help="easy-token eps source: anchor = target-eps reuse "
                        "(draft-free), draft = draft prediction")
    # selection
    p.add_argument("--selector", type=str, default="mask",
                   choices=["mask", "boundary", "freq", "delta", "combo",
                            "anchor", "random", "oracle"])
    p.add_argument("--hard_ratio", type=float, default=0.3,
                   help="fraction of tokens refreshed per sparse step; "
                        "0 = auto (cover the whole dilated hole)")
    p.add_argument("--restrict_to_mask", action="store_true",
                   help="spend the hard budget inside dilate(mask) first")
    p.add_argument("--mask_dilate", type=int, default=1,
                   help="token-grid dilation of the hole for routing")
    p.add_argument("--boundary_k", type=int, default=1)
    p.add_argument("--tok_thresh", type=float, default=0.5,
                   help="interior-fraction threshold for a hole token")
    p.add_argument("--block", type=int, default=1,
                   help="round hard set up to bxb token blocks (structured)")
    p.add_argument("--cw_mask", type=float, default=1.0)
    p.add_argument("--cw_bnd", type=float, default=1.0)
    p.add_argument("--cw_freq", type=float, default=0.5)
    p.add_argument("--cw_delta", type=float, default=0.5)
    # data / repro
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--box_prob", type=float, default=0.4)
    p.add_argument("--run_seed", type=int, default=0)
    p.add_argument("--mask_seed", type=int, default=1234)
    p.add_argument("--data_seed", type=int, default=7)
    p.add_argument("--no_reinject", action="store_true",
                   help="disable known-region latent re-injection (ablation)")
    p.add_argument("--save_debug_grid", action="store_true", default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.mode in ("mix", "draft") or args.easy == "draft" \
            or args.selector == "anchor":
        assert args.draft, "this configuration needs --draft"
    main(args)
