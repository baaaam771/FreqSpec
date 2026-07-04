#!/usr/bin/env python
"""
models/token_router.py — draft-only hard-token router g_psi(H_D, t, s).

The router predicts, per token, how much the target is needed — WITHOUT running
the target. It consumes only quantities the sparse system already has after the
draft forward:

    H_D,i   : draft final hidden token state           [B, N, d_draft]
    eps_d   : draft noise prediction (token stats)     6 scalars / token
    t       : timestep, sinusoidal embedding           t_dim
    (neigh) : 3x3 neighbor mean of |eps_d| token norm  1 scalar / token

Label design (key deviation from a fixed-threshold BCE): the sampler always
selects TopK tokens within one image at one timestep, so what matters is the
per-(image, timestep) RANKING of draft-target disagreement, and the raw
disagreement scale varies strongly with t (per-timestep normalization finding
in the One Verifier paper). The primary objective is therefore regression on
the per-(image, timestep) rank-normalized disagreement in [0,1]; an
asymmetric-BCE head at a fixed budget r is available as an option.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sinusoidal_t(t, dim=64):
    """t:[B] long -> [B, dim] sinusoidal embedding (same construction as DiT)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def token_scalar_feats(eps_d, p, hw):
    """Per-token scalar features from the draft prediction.
    eps_d: [B,C,H,W] -> [B, N, 7]:
        mean/std/absmean/max of per-pixel channel-mean eps within the token,
        token L2 norm, log-norm, 3x3 neighbor-mean of the token norm."""
    h, w = hw
    e = eps_d.mean(1, keepdim=True)                              # [B,1,H,W]
    mean = F.avg_pool2d(e, p, stride=p)
    sq = F.avg_pool2d(e.pow(2), p, stride=p)
    std = (sq - mean.pow(2)).clamp_min(0).sqrt()
    absmean = F.avg_pool2d(e.abs(), p, stride=p)
    mx = F.max_pool2d(e.abs(), p, stride=p)
    norm = F.avg_pool2d(eps_d.pow(2).mean(1, keepdim=True), p, stride=p).sqrt()
    lognorm = (norm + 1e-8).log()
    neigh = F.avg_pool2d(norm, 3, stride=1, padding=1)
    feats = torch.cat([mean, std, absmean, mx, norm, lognorm, neigh], dim=1)
    return feats.flatten(2).transpose(1, 2)                      # [B, N, 7]


N_SCALAR = 7


class TokenRouter(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear -> scalar per token."""
    def __init__(self, d_hidden_draft, t_dim=64, width=256):
        super().__init__()
        self.t_dim = t_dim
        d_in = d_hidden_draft + N_SCALAR + t_dim
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1))
        self.d_in = d_in

    def forward(self, h_draft, scal, t):
        """h_draft:[B,N,d], scal:[B,N,S], t:[B] -> logits [B,N]."""
        te = sinusoidal_t(t, self.t_dim)[:, None].expand(-1, h_draft.shape[1], -1)
        x = torch.cat([h_draft, scal, te], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def score(self, h_draft, scal, t):
        """Higher score = harder token (needs the target)."""
        return self.forward(h_draft, scal, t)


def build_router_from_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    r = TokenRouter(ck["d_hidden_draft"], ck.get("t_dim", 64), ck.get("width", 256))
    r.load_state_dict(ck["router"])
    r.to(device).eval()
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def rank_normalize(x, dim=-1):
    """Per-row rank in [0,1] (0 = smallest). Ties broken by argsort order."""
    order = x.argsort(dim=dim)
    ranks = torch.empty_like(order)
    ar = torch.arange(x.shape[dim], device=x.device).expand_as(order)
    ranks.scatter_(dim, order, ar)
    return ranks.float() / max(x.shape[dim] - 1, 1)


if __name__ == "__main__":
    r = TokenRouter(128)
    h = torch.randn(2, 256, 128); s = torch.randn(2, 256, N_SCALAR)
    t = torch.randint(0, 1000, (2,))
    print("router out", r(h, s, t).shape,
          "params", sum(p.numel() for p in r.parameters()) / 1e3, "k")
    print("rank_norm", rank_normalize(torch.randn(2, 5)))
