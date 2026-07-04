#!/usr/bin/env python
"""
models/dit.py — minimal class-conditional DiT (torch-only, no einops/diffusers).

Used for the FreqSpec token-grid PoC: a larger target DiT and a smaller draft DiT
predict eps in the same [B,C,H,W] space, so the existing FreqSpec verifier core
(which patchifies via avg_pool2d) operates on the patch-token grid with no change.
adaLN-Zero conditioning follows the original DiT. learn_sigma=False (eps only).

Configs:
    DiT-S    : dim=384 depth=12 heads=6   (~33M at patch 4, 32x32)
    DiT-Ti   : dim=192 depth=6  heads=3   (draft)
    DiT-Nano : dim=128 depth=4  heads=4   (smaller draft)
"""
import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        # sinusoidal emb is built in fp32; match the MLP weight dtype so the
        # model can be cast to bf16/fp16 for benchmarking (no-op in fp32)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """Class embedding with a null class for classifier-free guidance."""
    def __init__(self, num_classes, hidden, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, hidden)
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, y, train):
        if train and self.dropout > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout
            y = torch.where(drop, self.num_classes, y)
        return self.emb(y)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                       # [B, dim, h, w]
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (h, w)   # [B, N, dim]


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = self.ada(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), sh_msa, sc_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), sh_mlp, sc_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.lin(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=384, depth=12, heads=6,
                 num_classes=10, mlp_ratio=4.0, class_dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.patch = patch
        self.out_ch = in_ch
        self.x_embed = PatchEmbed(in_ch, dim, patch)
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = LabelEmbedder(num_classes, dim, class_dropout)
        gh = img_size // patch
        self.num_tokens = gh * gh
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, patch, self.out_ch)
        self._init()

    def _init(self):
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight); nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.lin.weight); nn.init.zeros_(self.final.lin.bias)

    def unpatchify(self, x, hw):
        h, w = hw
        p, c = self.patch, self.out_ch
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)           # B,C,h,p,w,p
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t, y, train=False):
        tok, hw = self.x_embed(x)
        tok = tok + self.pos
        c = self.t_embed(t) + self.y_embed(y, train)
        for blk in self.blocks:
            tok = blk(tok, c)
        tok = self.final(tok, c)                  # [B, N, p*p*out_ch]
        return self.unpatchify(tok, hw)           # [B, out_ch, H, W]


CONFIGS = {
    "DiT-B":    dict(dim=768, depth=12, heads=12),  # larger target (~130M at patch 4)
    "DiT-S":    dict(dim=384, depth=12, heads=6),
    "DiT-Ti":   dict(dim=192, depth=6,  heads=3),
    "DiT-Nano": dict(dim=128, depth=4,  heads=4),
}


def build_dit(name="DiT-S", **kw):
    cfg = dict(CONFIGS[name]); cfg.update(kw)
    return DiT(**cfg)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for n in CONFIGS:
        m = build_dit(n, img_size=32, patch=4, num_classes=10)
        x = torch.randn(2, 3, 32, 32); t = torch.randint(0, 1000, (2,)); y = torch.randint(0, 10, (2,))
        out = m(x, t, y)
        print(f"{n:9s} params={count_params(m)/1e6:6.2f}M  out={tuple(out.shape)}  "
              f"tokens={m.num_tokens}")
