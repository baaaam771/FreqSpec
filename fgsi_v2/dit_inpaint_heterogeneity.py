#!/usr/bin/env python
"""
dit_inpaint_heterogeneity.py — measure the DECIDING QUANTITY of the DACE
paper on the inpainting task: token-wise allocation beats uniform step
reduction only when BOTH factors are large,

    (i)  step-reduction sensitivity  : how much quality drops when the whole
         trajectory is shortened (measured in the MASK region, where errors
         matter for inpainting)
    (ii) spatial concentration       : how much of the target's temporal
         prediction change is carried by few tokens — and, critically for
         inpainting, whether that concentration ALIGNS WITH THE MASK

The DACE paper's forward prediction is that inpainting supplies factor (i)
(masked regions change fast, context barely moves under re-injection). This
script tests that prediction directly, BEFORE committing to the full sweep.

Outputs (JSON + npz):
    per-step and pooled: inside-mask vs outside-mask mean per-token change
        delta_i(t) = ||eps_i(z_t,t) - eps_i(z_t',t')||^2  (dense trajectory,
        known-region re-injection active, so "context barely moves" is
        actually realized rather than assumed)
    inside/outside change ratio (the mask-alignment index)
    top-30-percent token share of change, across-token CV, p90/p50
    fraction of top-r tokens that fall inside dilate(mask)  (does the mask
        already act as a free oracle router?)
    dense step-reduction curve of mask-MSE_t / mask-PSNR at
        S in {50,40,30,25,20,15,10}  (factor (i), mask-restricted)

Usage:
    python dit_inpaint_heterogeneity.py \
        --target ckpt_dit_inp/target.pt --target_model DiT-S-Inp \
        --data_root /mnt/HDD_12TB/bam_ki/datasets/datasets/imagenet64/val --n_traj 64 \
        --out results/dit_inp/heterogeneity
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit_inpaint import DiTInpaint, load_dit_inpaint
from training.scheduler import DDPMSchedule
from utils.inpaint_masks import (sample_masks, mask_to_tokens, dilate_tokens,
                                 known_latent)
from dit_inpaint_sampler import get_val_loader, per_token_sq
from utils.inpaint_metrics import region_psnr


@torch.no_grad()
def dense_traj_change(args, target, sch, x0, y, mask, dev, seed):
    """Run one dense S-step trajectory with re-injection; record per-token
    eps change between consecutive steps and the final image."""
    B = x0.shape[0]
    p = args.patch
    ts = sch.get_ddim_schedule_exact(args.steps)
    gen = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(B, 3, args.img_size, args.img_size, generator=gen,
                    device=dev)
    eps0 = torch.randn(B, 3, args.img_size, args.img_size, generator=gen,
                       device=dev)
    x_masked = x0 * (1 - mask)
    sac, s1m = sch.sqrt_alphas_cumprod, sch.sqrt_one_minus_alphas_cumprod
    z = mask * z + (1 - mask) * known_latent(x0, eps0, sac, s1m,
                                             int(ts[0].item()))
    deltas, eps_prev = [], None
    for i in range(len(ts)):
        t = int(ts[i].item())
        t_prev = int(ts[i + 1].item()) if i + 1 < len(ts) else -1
        tt = torch.full((B,), t, device=dev, dtype=torch.long)
        eps = target(DiTInpaint.pack(z, x_masked, mask), tt, y)
        if eps_prev is not None:
            deltas.append(per_token_sq(eps - eps_prev, p).cpu())   # [B,N]
        eps_prev = eps
        z, _ = sch.ddim_step(z, eps, t, t_prev)
        z = mask * z + (1 - mask) * known_latent(x0, eps0, sac, s1m, t_prev)
    x_out = (mask * z + (1 - mask) * x0).clamp(-1, 1)
    return torch.stack(deltas, 0), x_out                            # [S-1,B,N]


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    if os.path.exists(os.path.join(args.out, "heterogeneity.json")) and not args.overwrite:
        print(f"[skip] {args.out} already done (use --overwrite)"); return
    os.makedirs(args.out, exist_ok=True)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4,
                       beta_end=0.02, beta_schedule="linear", device=dev)
    target = load_dit_inpaint(args.target, args.target_model, args.img_size,
                              args.patch, args.num_classes, dev)
    args.n_samples = args.n_traj
    loader = get_val_loader(args)

    p = args.patch
    h = w = args.img_size // p

    all_delta, all_min, all_mdil = [], [], []
    ref50, per_S_out = {}, {S: [] for S in args.sweep}
    x0s, masks, ys = [], [], []
    for bi, (x0, y) in enumerate(loader):
        x0, y = x0.to(dev), y.to(dev)
        B = x0.shape[0]
        mask = sample_masks(B, args.img_size, args.img_size,
                            box_prob=0.4, seed=args.mask_seed + bi).to(dev)
        m_tok = (mask_to_tokens(mask, p) > 0.5).float()
        m_dil = dilate_tokens(m_tok, 1)

        d, x_ref = dense_traj_change(args, target, sch, x0, y, mask, dev,
                                     seed=args.run_seed + bi)
        all_delta.append(d)                                   # [S-1,B,N]
        all_min.append(m_tok.flatten(1).cpu())
        all_mdil.append(m_dil.flatten(1).cpu())
        ref50[bi] = x_ref
        x0s.append(x0); masks.append(mask); ys.append(y)

    delta = torch.cat(all_delta, 1)          # [S-1, Btot, N]
    m_in = torch.cat(all_min, 0)             # [Btot, N]
    m_dil = torch.cat(all_mdil, 0)
    S1, Btot, N = delta.shape

    # ---- (ii) concentration & mask alignment ----
    def _stats(dl):
        # dl: [Btot, N] one step (or pooled)
        mean_in = (dl * m_in).sum(1) / m_in.sum(1).clamp(min=1)
        out_m = 1 - m_in
        mean_out = (dl * out_m).sum(1) / out_m.sum(1).clamp(min=1)
        cv = dl.std(1) / dl.mean(1).clamp(min=1e-12)
        q = torch.quantile(dl, torch.tensor([0.5, 0.9]), dim=1)
        r = 0.3
        k = max(1, int(round(r * N)))
        top = torch.topk(dl, k, dim=1)
        share = top.values.sum(1) / dl.sum(1).clamp(min=1e-12)
        in_dil = torch.gather(m_dil, 1, top.indices).mean(1)
        return dict(mean_in=mean_in.mean().item(),
                    mean_out=mean_out.mean().item(),
                    in_out_ratio=(mean_in / mean_out.clamp(min=1e-12)).mean().item(),
                    cv=cv.mean().item(),
                    p90_p50=(q[1] / q[0].clamp(min=1e-12)).mean().item(),
                    top30_share=share.mean().item(),
                    top30_in_dilated_mask=in_dil.mean().item())

    per_step = [_stats(delta[s]) for s in range(S1)]
    pooled = _stats(delta.mean(0))

    # ---- (i) mask-restricted step-reduction sensitivity ----
    sens = {}
    for S in args.sweep:
        mse_t, psnr = 0.0, 0.0
        nb = 0
        for bi in range(len(x0s)):
            x0, mask, y = x0s[bi], masks[bi], ys[bi]
            args_S = args
            old_steps = args.steps
            args.steps = S
            _, x_out = dense_traj_change(args_S, target, sch, x0, y, mask,
                                         dev, seed=args.run_seed + bi)
            args.steps = old_steps
            x_ref = ref50[bi]
            m = mask
            mse_t += (((x_out - x_ref) ** 2 * m).flatten(1).sum(1)
                      / (m.flatten(1).sum(1) * 3 + 1e-8)).mean().item()
            psnr += region_psnr(x_out, x0, m).mean().item()
            nb += 1
        sens[S] = dict(mask_mse_t=mse_t / nb, mask_psnr=psnr / nb)
        print(f"[sens] S={S}: mask_mse_t={sens[S]['mask_mse_t']:.5f} "
              f"mask_psnr={sens[S]['mask_psnr']:.3f}")

    res = dict(config=vars(args), pooled=pooled, per_step=per_step,
               step_reduction_sensitivity=sens,
               interpretation=dict(
                   factor_ii_alignment="in_out_ratio much greater than 1 and "
                       "top30_in_dilated_mask near 1 mean the mask IS the "
                       "router — factor (ii) holds and is free",
                   factor_i="mask_mse_t rising steeply as S drops means the "
                       "model is past its step-reduction knee in the hole — "
                       "factor (i) holds; flat curve reproduces the "
                       "ImageNet-64 generation null result"))
    with open(os.path.join(args.out, "heterogeneity.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(args.out, "delta_tokens.npz"),
             delta=delta.numpy(), mask_in=m_in.numpy(),
             mask_dil=m_dil.numpy())
    print(json.dumps(dict(pooled=pooled), indent=2))
    print(f"[done] {args.out}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--target_model", type=str, default="DiT-S-Inp")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--steps", type=int, default=50,
                   help="dense trajectory length for the change measurement")
    p.add_argument("--sweep", type=int, nargs="+",
                   default=[50, 40, 30, 25, 20, 15, 10])
    p.add_argument("--n_traj", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--run_seed", type=int, default=0)
    p.add_argument("--mask_seed", type=int, default=1234)
    p.add_argument("--data_seed", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
