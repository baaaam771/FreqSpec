#!/usr/bin/env python
"""
models/dit_inpaint.py — mask-conditioned class-conditional DiT for ImageNet-64
inpainting (pixel space), built directly on models/dit.py so that EVERY
function in models/dit_sparse.py (dense-with-cache, cached sparse suffix,
FLOPs accounting) works unchanged.

Input  (in_ch = 7):  [ z_t (3) | x_masked (3) | mask (1) ]
Output (out_ch = 3): eps prediction for z_t

The only structural deviation from DiT is out_ch != in_ch, which requires
rebuilding FinalLayer. All attributes consumed by dit_sparse.py
(patch, out_ch, pos, blocks, final, x_embed, t_embed, y_embed, unpatchify,
num_tokens) are preserved, so:

    dit_forward_dense_with_cache(model, x_in, t, y, m)      # anchors
    sparse_target_eps_cached(model, x_in, t, y, idx, m, canvas, cache)
    dit_model_flops(model, mode, m, k)

all operate on a DiTInpaint by passing the pre-concatenated 7-channel x_in.

Configs mirror models/dit.py:
    DiT-S-Inp    : dim=384 depth=12 heads=6   (target, ~33M)
    DiT-Ti-Inp   : dim=192 depth=6  heads=3   (draft)
    DiT-Nano-Inp : dim=128 depth=4  heads=4   (smaller draft)
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dit import DiT, FinalLayer, CONFIGS, count_params  # noqa: E402


class DiTInpaint(DiT):
    """DiT with decoupled in_ch/out_ch. in_ch=7 (z_t + masked image + mask),
    out_ch=3 (eps). Everything else identical to DiT, so dit_sparse.py
    primitives (which read model.patch / model.out_ch / model.final) work."""

    def __init__(self, img_size=64, patch=4, in_ch=7, out_ch=3, dim=384,
                 depth=12, heads=6, num_classes=1000, mlp_ratio=4.0,
                 class_dropout=0.1):
        super().__init__(img_size=img_size, patch=patch, in_ch=in_ch, dim=dim,
                         depth=depth, heads=heads, num_classes=num_classes,
                         mlp_ratio=mlp_ratio, class_dropout=class_dropout)
        # DiT sets out_ch = in_ch and builds FinalLayer accordingly; rebuild.
        self.out_ch = out_ch
        self.final = FinalLayer(dim, patch, out_ch)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.lin.weight)
        nn.init.zeros_(self.final.lin.bias)

    @staticmethod
    def pack(z_t, x_masked, mask):
        """[B,3,H,W], [B,3,H,W], [B,1,H,W] -> [B,7,H,W]."""
        return torch.cat([z_t, x_masked, mask], dim=1)

    def forward_inpaint(self, z_t, t, y, x_masked, mask, train=False):
        """Convenience wrapper; the plain forward(x_in, t, y) is what the
        sparse-execution primitives call, so both entry points exist."""
        return self.forward(self.pack(z_t, x_masked, mask), t, y, train=train)


INPAINT_CONFIGS = {
    "DiT-S-Inp":    dict(CONFIGS["DiT-S"]),
    "DiT-Ti-Inp":   dict(CONFIGS["DiT-Ti"]),
    "DiT-Nano-Inp": dict(CONFIGS["DiT-Nano"]),
    "DiT-B-Inp":    dict(CONFIGS["DiT-B"]),
}


def build_dit_inpaint(name="DiT-S-Inp", **kw):
    cfg = dict(INPAINT_CONFIGS[name]); cfg.update(kw)
    return DiTInpaint(**cfg)


def load_dit_inpaint(path, name, img_size, patch, num_classes, dev):
    m = build_dit_inpaint(name, img_size=img_size, patch=patch,
                          num_classes=num_classes, class_dropout=0.0).to(dev).eval()
    ck = torch.load(path, map_location=dev)
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m


if __name__ == "__main__":
    # sanity: shapes + dit_sparse compatibility (dense suffix == forward)
    from models.dit_sparse import (dit_forward_dense_with_cache,
                                   sparse_target_eps_cached, topk_index,
                                   dit_model_flops)
    torch.manual_seed(0)
    m = build_dit_inpaint("DiT-Nano-Inp", img_size=64, patch=4,
                          num_classes=1000, class_dropout=0.0).eval()
    B = 2
    z = torch.randn(B, 3, 64, 64)
    xm = torch.randn(B, 3, 64, 64)
    mk = (torch.rand(B, 1, 64, 64) > 0.5).float()
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 1000, (B,))
    x_in = DiTInpaint.pack(z, xm, mk)
    with torch.no_grad():
        ref = m(x_in, t, y)
        eps_a, cache = dit_forward_dense_with_cache(m, x_in, t, y, m=0)
        # fresh-cache exactness at any sparsity (DACE guarantee)
        idx = torch.sort(topk_index(torch.randn(B, m.num_tokens), 0.3),
                         dim=1).values
        eps_sp = sparse_target_eps_cached(m, x_in, t, y, idx, 0, ref.clone(),
                                          cache)
    print("params", count_params(m) / 1e6, "M  out", tuple(ref.shape))
    print("dense-with-cache max|d|", (ref - eps_a).abs().max().item())
    print("fresh-cache sparse max|d|", (ref - eps_sp).abs().max().item())
    print("sparse_attn r=0.3 MAC ratio",
          dit_model_flops(m, "sparse_attn", m=0, k=int(0.3 * m.num_tokens))
          / dit_model_flops(m, "dense"))
