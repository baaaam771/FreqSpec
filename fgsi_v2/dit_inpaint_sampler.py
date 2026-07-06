#!/usr/bin/env python
"""
dit_inpaint_sampler.py — DACE (depth-aligned cached execution) on DiT
INPAINTING (ImageNet-64, pixel space). REVISED per the reviewer notes.

Changes vs the first version (review item numbers in brackets):
  [1]  frequency score from the anchor's predicted x0 (or known image), not
       only noisy z_t:  --freq_src {zt, x0_anchor, known}  (x0_anchor default)
  [2]  explicit selection region:  --region {global, mask}. global scores ALL
       tokens (random/freq/delta/oracle); mask confines the budget to the hole
       so the mask prior and the within-hole ranking are separable.
  [3]  EXACT per-sample mask budget (--budget mask_exact): each sample refreshes
       exactly its own hole tokens (padded gather, true k_b charged) instead of
       a batch-max auto budget that leaked outside small masks.
  [4]  selection boundary width (--selection_boundary_k) separate from the
       execution mask dilation (--execution_mask_dilate).
  [5]  combo rank-normalizes freq/delta before the weighted sum.
  [6]  block-STRUCTURED selection is named as such; execution stays
       gather/scatter and the TRUE block-covered k is charged.
  [7]  output-vs-dense50 distances named *_to_dense50 (not LPIPS_t/MSE_t).
  [8]  known-region PSNR/SSIM + boundary-ring LPIPS reported.
  [12] per-sample records -> mask-size bucketing (small/medium/large).
  [14] --suffix {cache, frozen} crossed with --no_reinject gives the 4-way
       ablation isolating the depth-aligned cache from re-injection.

The dense / reduced-step REFERENCE path is unchanged and ALWAYS re-injects
(force_reinject=True), so the Stage-1 dense-50 reference cache in --ref_dir
stays valid and comparable.
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

from models.dit_inpaint import DiTInpaint, load_dit_inpaint, build_dit_inpaint
from models.dit_sparse import (dit_forward_dense_with_cache,
                               sparse_target_eps, sparse_target_eps_cached,
                               dit_model_flops, topk_index)
from training.scheduler import DDPMSchedule
from utils.inpaint_masks import (sample_masks, mask_to_tokens, dilate_tokens,
                                 boundary_band, pixel_boundary_ring,
                                 combo_score, block_round_indices,
                                 exact_set_indices, topk_within, known_latent)
from utils.inpaint_metrics import (region_mse, region_psnr, region_ssim,
                                   region_lpips, mask_size_bucket)

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


class Timer:
    def __init__(self, dev):
        self.cuda = dev.type == "cuda"; self.acc = {}

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


def hf_energy(img, h, w, dwt):
    if dwt is not None:
        _, lh, hl, hh = dwt(img)
        e = (lh ** 2 + hl ** 2 + hh ** 2).mean(1, keepdim=True)
    else:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=img.device).view(1, 1, 3, 3).repeat(
            img.shape[1], 1, 1, 1)
        e = F.conv2d(img, k, padding=1, groups=img.shape[1]).pow(2).mean(1, keepdim=True)
    return F.adaptive_avg_pool2d(e, (h, w)).flatten(1)


def per_token_sq(diff, p):
    return F.avg_pool2d(diff.pow(2).mean(1, keepdim=True), p, stride=p).flatten(1)


def x0_from_eps(z, eps, t, sac, s1m):
    return (z - s1m[t] * eps) / sac[t]


# --------------------------------------------------------------- selector core
@torch.no_grad()
def selector_raw(args, st):
    sel, B, N = args.selector, st["B"], st["N"]
    h, w = st["hw"]; dev = st["z"].device
    m_tok = st["mask_tok_dil"]; b_tok = st["bnd_tok"]
    if sel == "mask":
        return m_tok + 0.5 * b_tok + 1e-4 * torch.rand(B, N, device=dev)
    if sel == "boundary":
        return b_tok + 1e-4 * torch.rand(B, N, device=dev)
    if sel == "freq":
        return st["freq_tok"]
    if sel == "delta":
        return st["delta_tok"] if st["delta_tok"] is not None else m_tok + 0.5 * b_tok
    if sel == "combo":
        return combo_score(M_tok=m_tok, B_tok=b_tok, F_tok=st["freq_tok"],
                           D_tok=st["delta_tok"], a=args.cw_mask, b=args.cw_bnd,
                           g=args.cw_freq, d=args.cw_delta)
    if sel == "anchor":
        return st["anchor_dis_tok"] if st["anchor_dis_tok"] is not None else m_tok + 0.5 * b_tok
    if sel == "random":
        return torch.rand(B, N, device=dev)
    if sel == "oracle":
        eps_t_dense = st["target"](st["x_in"], st["tt"], st["y"])
        st["oracle_extra_nfe"] += 1
        return per_token_sq(st["eps_easy_now"] - eps_t_dense, args.patch)
    raise ValueError(f"unknown selector {sel}")


@torch.no_grad()
def choose_hard(args, st):
    """(idx [B,kpad], true_k [B]) honoring --region and --budget."""
    score = selector_raw(args, st)
    h, w, N = st["hw"][0], st["hw"][1], st["N"]
    if args.region == "mask":
        # strict hole eligibility — NO boundary leak regardless of cw_bnd
        elig = (st["mask_tok_dil"] > 0.5).float()
    elif args.region == "mask_plus_boundary":
        elig = ((st["mask_tok_dil"] + st["bnd_tok"]) > 0.5).float()
    else:  # global
        elig = torch.ones(st["B"], N, device=score.device)

    if args.budget == "mask_exact":
        idx, cnt = exact_set_indices(elig)
    else:
        if args.region in ("mask", "mask_plus_boundary"):
            idx, cnt = topk_within(score, elig, args.hard_ratio)
        else:
            k = max(1, int(round(args.hard_ratio * N)))
            idx = torch.sort(topk_index(score, k / N), dim=1).values
            cnt = torch.full((st["B"],), idx.shape[1], device=score.device)
    if args.block > 1:
        idx, cnt = block_round_indices(idx, h, w, args.block)
    return idx, cnt


# ------------------------------------------------------------------- sampling
@torch.no_grad()
def run_batch(args, target, draft, sch, ts, x0, y, mask, gen, dev, timer,
              dense_only_steps=None, force_reinject=False):
    B = x0.shape[0]; p = args.patch
    h = w = args.img_size // p; N = h * w
    reinject = True if force_reinject else (not args.no_reinject)

    x_masked = x0 * (1 - mask)
    z = torch.randn(B, 3, args.img_size, args.img_size, generator=gen, device=dev)
    eps0 = torch.randn(B, 3, args.img_size, args.img_size, generator=gen, device=dev)
    sac = sch.sqrt_alphas_cumprod; s1m = sch.sqrt_one_minus_alphas_cumprod

    schedule = ts if dense_only_steps is None else sch.get_ddim_schedule_exact(dense_only_steps)
    mode = "dense" if dense_only_steps is not None else args.mode

    m_tok_bin = (mask_to_tokens(mask, p) > args.tok_thresh).float()
    m_tok_dil = dilate_tokens(m_tok_bin, args.execution_mask_dilate)
    bnd = boundary_band(m_tok_bin, args.selection_boundary_k)
    dwt = DWT2D("haar").to(dev) if _HAS_DWT else None
    known_hf = hf_energy(x_masked, h, w, dwt)

    st = dict(B=B, N=N, hw=(h, w), target=target, y=y, dwt=dwt,
              mask_tok_dil=m_tok_dil.flatten(1), bnd_tok=bnd.flatten(1),
              delta_tok=None, anchor_dis_tok=None, freq_tok=known_hf,
              oracle_extra_nfe=0)
    macs_dense_t = dit_model_flops(target, "dense")
    macs_dense_d = dit_model_flops(draft, "dense") if draft is not None else 0
    stats = dict(tgt_dense_nfe=0, tgt_sparse_nfe=0, drf_nfe=0,
                 macs=0.0, exec_macs=0.0, sum_true_k=0.0, sum_exec_k=0.0,
                 n_sparse=0, oracle_extra_nfe=0)

    t0 = int(schedule[0].item())
    if reinject:
        z = mask * z + (1 - mask) * known_latent(x0, eps0, sac, s1m, t0)

    cache, eps_anchor = None, None
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
                eps, cache = dit_forward_dense_with_cache(target, x_in, tt, y, args.split_m)
                if eps_anchor is not None:
                    st["delta_tok"] = per_token_sq(eps - eps_anchor, p)
                eps_anchor = eps
                if args.freq_src == "x0_anchor":
                    st["freq_tok"] = hf_energy(x0_from_eps(z, eps, t, sac, s1m), h, w, dwt)
                elif args.freq_src == "zt":
                    st["freq_tok"] = hf_energy(z, h, w, dwt)
                if args.selector == "anchor":
                    eps_d = draft(x_in, tt, y)
                    stats["drf_nfe"] += 1; stats["macs"] += macs_dense_d; stats["exec_macs"] += macs_dense_d
                    st["anchor_dis_tok"] = per_token_sq(eps_d - eps, p)
            else:
                eps = target(x_in, tt, y)
            timer.stop("target_dense")
            stats["tgt_dense_nfe"] += 1; stats["macs"] += macs_dense_t; stats["exec_macs"] += macs_dense_t

        elif mode == "dace":
            if args.easy == "anchor":
                eps_easy = eps_anchor
            else:
                timer.start(); eps_easy = draft(x_in, tt, y); timer.stop("draft")
                stats["drf_nfe"] += 1; stats["macs"] += macs_dense_d; stats["exec_macs"] += macs_dense_d
            st["eps_easy_now"] = eps_easy
            if args.freq_src == "zt":
                st["freq_tok"] = hf_energy(z, h, w, dwt)
            timer.start(); idx, cnt = choose_hard(args, st); timer.stop("select")
            timer.start()
            if args.suffix == "cache":
                eps = sparse_target_eps_cached(target, x_in, tt, y, idx,
                                               args.split_m, eps_easy.clone(), cache)
            else:
                eps = sparse_target_eps(target, x_in, tt, y, idx, args.split_m,
                                        "sparse_attn", eps_easy.clone(),
                                        refresh_every=args.frozen_refresh)
            timer.stop("target_sparse")
            k_true = float(cnt.float().mean().item())   # ideal sample-wise k
            k_exec = int(idx.shape[1])                   # padded rectangular k
            rf = args.frozen_refresh if args.suffix == "frozen" else 0
            stats["tgt_sparse_nfe"] += 1; stats["n_sparse"] += 1
            stats["sum_true_k"] += k_true; stats["sum_exec_k"] += k_exec
            stats["macs"] += dit_model_flops(target, "sparse_attn",
                                             m=args.split_m,
                                             k=max(1, int(round(k_true))),
                                             refresh_every=rf)
            stats["exec_macs"] += dit_model_flops(target, "sparse_attn",
                                                  m=args.split_m, k=k_exec,
                                                  refresh_every=rf)

        elif mode == "mix":
            from models.dit_sparse import patchify_img, scatter_tokens, gather_tokens
            timer.start(); eps_t = target(x_in, tt, y); eps_d = draft(x_in, tt, y)
            timer.stop("dense_both")
            stats["tgt_dense_nfe"] += 1; stats["drf_nfe"] += 1
            stats["macs"] += macs_dense_t + macs_dense_d; stats["exec_macs"] += macs_dense_t + macs_dense_d
            dis = per_token_sq(eps_d - eps_t, p)
            k = max(1, int(round(args.hard_ratio * N)))
            idx = torch.sort(topk_index(dis, k / N), dim=1).values
            cv = patchify_img(eps_d, p); tv = patchify_img(eps_t, p)
            cv = scatter_tokens(cv, idx, gather_tokens(tv, idx))
            eps = target.unpatchify(cv, (h, w))
            stats["sum_true_k"] += k; stats["sum_exec_k"] += k; stats["n_sparse"] += 1

        elif mode == "draft":
            timer.start(); eps = draft(x_in, tt, y); timer.stop("draft")
            stats["drf_nfe"] += 1; stats["macs"] += macs_dense_d; stats["exec_macs"] += macs_dense_d
        else:
            raise ValueError(mode)

        timer.start()
        z, _ = sch.ddim_step(z, eps, t, t_prev)
        if reinject:
            z = mask * z + (1 - mask) * known_latent(x0, eps0, sac, s1m, t_prev)
        timer.stop("scheduler")

    stats["oracle_extra_nfe"] = st["oracle_extra_nfe"]
    x_model = z.clamp(-1, 1)
    x_paste = (mask * z + (1 - mask) * x0).clamp(-1, 1)
    return x_paste, x_model, stats


def _agg(recs):
    if not recs:
        return {}
    keys = ["coverage", "mask_psnr", "known_psnr", "known_ssim",
            "mask_mse_to_dense50", "mask_lpips", "mask_lpips_to_dense50",
            "boundary_lpips_to_dense50"]
    out = {"n": len(recs)}
    for k in keys:
        vals = [r[k] for r in recs if r[k] is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
    return out


# ----------------------------------------------------------------------- main
@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png"); os.makedirs(png_dir, exist_ok=True)
    if args.ref_dir:
        os.makedirs(args.ref_dir, exist_ok=True)

    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    ts = sch.get_ddim_schedule_exact(args.steps)
    target = load_dit_inpaint(args.target, args.target_model, args.img_size,
                              args.patch, args.num_classes, dev)
    draft = load_dit_inpaint(args.draft, args.draft_model, args.img_size,
                             args.patch, args.num_classes, dev) if args.draft else None
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev).eval() if _HAS_LPIPS else None

    loader = get_val_loader(args); timer = Timer(dev)
    records = []
    comp = dict(macs=0.0, exec_macs=0.0, tgt_dense_nfe=0.0, tgt_sparse_nfe=0.0,
                drf_nfe=0.0, sum_true_k=0.0, sum_exec_k=0.0, n_sparse=0.0,
                oracle_extra_nfe=0.0, nb=0)
    macs_ref = 50 * dit_model_flops(target, "dense")

    from torchvision.utils import save_image
    img_id = 0
    for bi, (x0, y) in enumerate(loader):
        x0, y = x0.to(dev), y.to(dev); B = x0.shape[0]
        mask = sample_masks(B, args.img_size, args.img_size,
                            box_prob=args.box_prob, seed=args.mask_seed + bi).to(dev)
        ring = pixel_boundary_ring(mask, ring=args.boundary_ring)

        ref_path = os.path.join(args.ref_dir or args.out_dir,
                                f"ref_b{bi}_s{args.run_seed}_m{args.mask_seed}"
                                f"_n{args.n_samples}_d{args.data_seed}.pt")
        if os.path.exists(ref_path):
            x_ref = torch.load(ref_path, map_location=dev)
        else:
            gr = torch.Generator(device=dev).manual_seed(args.run_seed + bi)
            x_ref, _, _ = run_batch(args, target, draft, sch, ts, x0, y, mask,
                                    gr, dev, Timer(dev), dense_only_steps=50,
                                    force_reinject=True)
            torch.save(x_ref.cpu(), ref_path); x_ref = x_ref.to(dev)

        gen = torch.Generator(device=dev).manual_seed(args.run_seed + bi)
        x_paste, x_model, stats = run_batch(args, target, draft, sch, ts, x0,
                                            y, mask, gen, dev, timer)
        for k in ("macs", "exec_macs", "tgt_dense_nfe", "tgt_sparse_nfe",
                  "drf_nfe", "sum_true_k", "sum_exec_k", "n_sparse",
                  "oracle_extra_nfe"):
            comp[k] += stats[k]
        comp["nb"] += 1

        cov = mask.flatten(1).mean(1)
        m_psnr = region_psnr(x_paste, x0, mask)
        k_psnr = region_psnr(x_model, x0, 1 - mask)
        k_ssim = region_ssim(x_model, x0, 1 - mask)
        m_mse_ref = region_mse(x_paste, x_ref, mask)
        m_lp = region_lpips(lpips_fn, x_paste, x0, mask)
        m_lp_ref = region_lpips(lpips_fn, x_paste, x_ref, mask)
        b_lp_ref = region_lpips(lpips_fn, x_paste, x_ref, ring)
        buckets = mask_size_bucket(cov, args.bucket_small, args.bucket_large)
        for j in range(B):
            records.append(dict(
                coverage=cov[j].item(), mask_psnr=m_psnr[j].item(),
                known_psnr=k_psnr[j].clamp(max=99).item(),
                known_ssim=k_ssim[j].item(),
                mask_mse_to_dense50=m_mse_ref[j].item(),
                mask_lpips=(m_lp[j].item() if m_lp is not None else None),
                mask_lpips_to_dense50=(m_lp_ref[j].item() if m_lp_ref is not None else None),
                boundary_lpips_to_dense50=(b_lp_ref[j].item() if b_lp_ref is not None else None),
                bucket=buckets[j]))
            save_image(x_paste[j] * 0.5 + 0.5,
                       os.path.join(png_dir, f"{img_id:06d}.png")); img_id += 1
        if bi == 0 and args.save_debug_grid:
            dbg = torch.cat([x0[:8] * (1 - mask[:8]) - mask[:8], x_ref[:8],
                             x_paste[:8], x0[:8]], 0)
            save_image(dbg * 0.5 + 0.5,
                       os.path.join(args.out_dir, "debug_grid.png"), nrow=8)
        print(f"[batch {bi}] {len(records)}/{args.n_samples}")

    nb = max(1, comp["nb"])
    ns = comp["n_sparse"] if comp["n_sparse"] else 1
    compute = dict(
        ideal_macs_per_image=comp["macs"] / nb,
        ideal_macs_vs_dense50=(comp["macs"] / nb) / macs_ref,
        executed_macs_per_image=comp["exec_macs"] / nb,
        executed_macs_vs_dense50=(comp["exec_macs"] / nb) / macs_ref,
        tgt_dense_nfe=comp["tgt_dense_nfe"] / nb,
        tgt_sparse_nfe=comp["tgt_sparse_nfe"] / nb,
        drf_nfe=comp["drf_nfe"] / nb,
        mean_true_k=comp["sum_true_k"] / ns,
        mean_executed_k=comp["sum_exec_k"] / ns,
        oracle_extra_nfe=comp["oracle_extra_nfe"] / nb,
        delta_warmup_policy="mask_boundary_fallback")

    res = dict(config=vars(args), overall=_agg(records),
               by_bucket={b: _agg([r for r in records if r["bucket"] == b])
                          for b in ("small", "medium", "large")},
               compute=compute, wallclock_ms=dict(timer.acc),
               lpips_available=_HAS_LPIPS, dwt_available=_HAS_DWT)
    with open(os.path.join(args.out_dir, "records.json"), "w") as f:
        json.dump(records, f)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"overall": res["overall"], "compute": compute}, indent=2))
    print(f"[done] {args.out_dir}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--target_model", type=str, default="DiT-S-Inp")
    p.add_argument("--draft", type=str, default="")
    p.add_argument("--draft_model", type=str, default="DiT-Nano-Inp")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ref_dir", type=str, default="")
    p.add_argument("--mode", type=str, default="dace",
                   choices=["dense", "dace", "mix", "draft"])
    p.add_argument("--suffix", type=str, default="cache",
                   choices=["cache", "frozen"],
                   help="cache = DACE depth-aligned cache; frozen = "
                        "depth-mismatched frozen context (ablation item 14)")
    p.add_argument("--frozen_refresh", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cache_period", type=int, default=2)
    p.add_argument("--split_m", type=int, default=0)
    p.add_argument("--easy", type=str, default="anchor", choices=["anchor", "draft"])
    p.add_argument("--region", type=str, default="mask",
                   choices=["global", "mask", "mask_plus_boundary"])
    p.add_argument("--budget", type=str, default="ratio",
                   choices=["ratio", "mask_exact"])
    p.add_argument("--selector", type=str, default="mask",
                   choices=["mask", "boundary", "freq", "delta", "combo",
                            "anchor", "random", "oracle"])
    p.add_argument("--freq_src", type=str, default="x0_anchor",
                   choices=["zt", "x0_anchor", "known"])
    p.add_argument("--hard_ratio", type=float, default=0.3,
                   help="ratio of the ELIGIBLE set (region-dependent)")
    p.add_argument("--execution_mask_dilate", type=int, default=1)
    p.add_argument("--selection_boundary_k", type=int, default=1)
    p.add_argument("--tok_thresh", type=float, default=0.5)
    p.add_argument("--block", type=int, default=1,
                   help="block-structured selection (2 = 2x2); execution stays "
                        "gather/scatter, true k charged")
    p.add_argument("--cw_mask", type=float, default=1.0)
    p.add_argument("--cw_bnd", type=float, default=1.0)
    p.add_argument("--cw_freq", type=float, default=0.5)
    p.add_argument("--cw_delta", type=float, default=0.5)
    p.add_argument("--boundary_ring", type=int, default=6)
    p.add_argument("--bucket_small", type=float, default=0.10)
    p.add_argument("--bucket_large", type=float, default=0.25)
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
                   help="disable known-region re-injection (ablation item 14; "
                        "the reference still re-injects)")
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
