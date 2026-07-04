#!/usr/bin/env python
"""
dit_router_eval.py — Stage 8: evaluate the draft-only router before deploying it.

On fresh dense-target trajectories (never seen in router training if a
different --seed is used) it compares selectors by:

    per-timestep AURC        (paper's primary reliability metric; risk =
                              per-timestep rank-normalized eps disagreement)
    hard-token recall at r   in {0.1, 0.3, 0.5, 0.7} (oracle TopK as truth)
    false-accept rate at r   (fraction of accepted (=easy-classified) tokens
                              that are oracle-hard)
    Spearman vs oracle
    router forward latency   (ms per step, batch --batch)

Selectors: random / frequency / norm(draft-eps) / router / oracle.

Usage:
    python dit_router_eval.py \
        --target ... --draft ... --router ckpt_dit_in64/router_nano.pt \
        --img_size 64 --patch 4 --num_classes 1000 \
        --n_traj 64 --steps 50 --seed 1234 \
        --out_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/router_eval
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dit_sparse import dit_forward_tokens
from models.token_router import (token_scalar_feats, build_router_from_ckpt,
                                 rank_normalize)
from models.wavelet import DWT2D, lwd_wavelet_saliency
from training.scheduler import DDPMSchedule
from dit_token_sampler import load_dit

RATIOS = (0.1, 0.3, 0.5, 0.7)


def aurc_per_step(score, risk, n_cov=19):
    """score, risk: [B,N] for ONE timestep. Higher score = accept target first
    is inverted here: we ACCEPT the draft on the LOWEST-score tokens. AURC over
    coverage grid; risk is rank-normalized per row."""
    r = rank_normalize(risk)
    covs = torch.linspace(0.05, 0.95, n_cov, device=score.device)
    B, N = score.shape
    order = score.argsort(dim=1)                     # ascending: easiest first
    r_sorted = torch.gather(r, 1, order)
    csum = r_sorted.cumsum(dim=1)
    risks = []
    for c in covs:
        k = max(1, int(round(float(c) * N)))
        risks.append((csum[:, k - 1] / k).mean().item())
    return float(np.trapezoid(risks, covs.cpu().numpy()))


@torch.no_grad()
def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    target = load_dit(args.target, args.target_model, args, dev)
    draft = load_dit(args.draft, args.draft_model, args, dev)
    router = build_router_from_ckpt(args.router, dev)
    dwt = DWT2D("haar").to(dev)
    p = args.patch
    hw = (args.img_size // p, args.img_size // p)

    ts = sch.get_ddim_schedule_exact(args.steps).tolist()
    g = torch.Generator(device=dev).manual_seed(args.seed)

    sel_names = ["random", "frequency", "norm", "router", "oracle"]
    aurcs = {s: [] for s in sel_names}
    rec = {s: {r: [] for r in RATIOS} for s in sel_names}
    far = {s: {r: [] for r in RATIOS} for s in sel_names}
    spear = []
    router_ms = []

    n_done = 0
    while n_done < args.n_traj:
        b = min(args.batch, args.n_traj - n_done)
        y = torch.randint(0, args.num_classes, (b,), device=dev, generator=g)
        z = torch.randn(b, 3, args.img_size, args.img_size, device=dev, generator=g)
        for i, t in enumerate(ts):
            t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
            tt = torch.full((b,), int(t), device=dev, dtype=torch.long)
            eps_t = target(z, tt, y)
            eps_d, h_d = dit_forward_tokens(draft, z, tt, y)
            d = F.avg_pool2d((eps_d - eps_t).pow(2).mean(1, keepdim=True),
                             p, stride=p).flatten(1)              # risk [B,N]
            scal = token_scalar_feats(eps_d, p, hw)

            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s_router = router(h_d.float(), scal.float(), tt)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            router_ms.append((time.perf_counter() - t0) * 1e3)

            scores = dict(
                random=torch.rand_like(d),
                frequency=lwd_wavelet_saliency(z, dwt, target_size=hw).flatten(1),
                norm=F.avg_pool2d(eps_d.pow(2).mean(1, keepdim=True),
                                  p, stride=p).flatten(1),
                router=s_router,
                oracle=d)
            N = d.shape[1]
            true_masks = {}
            for r in RATIOS:
                k = max(1, int(round(r * N)))
                true_masks[r] = torch.zeros_like(d, dtype=torch.bool).scatter_(
                    1, d.topk(k, dim=1).indices, True)
            for s, sc in scores.items():
                aurcs[s].append(aurc_per_step(sc, d))
                for r in RATIOS:
                    k = max(1, int(round(r * N)))
                    pm = torch.zeros_like(d, dtype=torch.bool).scatter_(
                        1, sc.topk(k, dim=1).indices, True)
                    tm = true_masks[r]
                    rec[s][r].append(((pm & tm).sum(1).float() / k).mean().item())
                    # accepted = NOT selected for target; FA = accepted but hard
                    acc = ~pm
                    far[s][r].append(((acc & tm).sum(1).float()
                                      / acc.sum(1).clamp_min(1)).mean().item())
            rr, dr = rank_normalize(s_router), rank_normalize(d)
            rrc = rr - rr.mean(1, keepdim=True); drc = dr - dr.mean(1, keepdim=True)
            spear.append(((rrc * drc).sum(1) /
                          (rrc.norm(dim=1) * drc.norm(dim=1) + 1e-8)).mean().item())
            z, _ = sch.ddim_step(z, eps_t, int(t), t_prev, eta=0.0)
        n_done += b
        print(f"[router-eval] {n_done}/{args.n_traj}")

    out = dict(router=args.router, n_traj=args.n_traj, steps=args.steps,
               router_ms_per_step=round(float(np.mean(router_ms)), 3),
               router_spearman=round(float(np.mean(spear)), 4),
               selectors={})
    for s in sel_names:
        out["selectors"][s] = dict(
            per_timestep_aurc=round(float(np.mean(aurcs[s])), 5),
            hard_recall={r: round(float(np.mean(rec[s][r])), 4) for r in RATIOS},
            far={r: round(float(np.mean(far[s][r])), 4) for r in RATIOS})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "router_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[router-eval] wrote {os.path.join(args.out_dir, 'router_eval.json')}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--draft", type=str, required=True)
    ap.add_argument("--router", type=str, required=True)
    ap.add_argument("--target_model", type=str, default="DiT-S")
    ap.add_argument("--draft_model", type=str, default="DiT-Nano")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--n_traj", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234,
                    help="use a different seed than dataset collection")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
