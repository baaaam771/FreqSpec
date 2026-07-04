#!/usr/bin/env python
"""
models/dit_sparse.py — sparse target execution over an existing (frozen) DiT.

This module adds NO parameters and requires NO retraining: every function
operates on a stock models.dit.DiT instance, so existing checkpoints load
unchanged. It implements the "Verifier -> Router -> Sparse execution" roadmap:

    dense prefix  T_{1:m}        : all tokens, standard blocks
    sparse suffix T_{m+1:L}      : only hard tokens receive updates

Suffix modes
    "dense"      : reference (full blocks; sanity check, max |delta|=0 vs DiT.forward)
    "sparse_mlp" : attention over ALL tokens (context preserved), MLP update only
                   on hard tokens. Low-risk first step of sparse execution.
    "sparse_attn": hard tokens are the only queries; all tokens serve as K/V
                   (easy-token K/V progressively frozen at their last updated
                   state). MLP also hard-only. Maximum FLOPs reduction.

Easy-token handling: easy tokens keep their running hidden state ("identity",
method A in the roadmap). Their final eps is NEVER taken from the sparse target;
the caller substitutes the draft prediction there, matching the verifier
formulation (easy tokens' only role inside the suffix is attention context).

FinalLayer is also evaluated on hard tokens only — easy-token outputs are
discarded anyway, so this is a free saving.

All functions assume batch_first token layout [B, N, D] and per-image fixed
hard-token count k (rectangular gather => GPU-friendly, no ragged tensors).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dit import modulate  # noqa: E402


# ---------------------------------------------------------------- token utils
def patchify_img(x, p):
    """Inverse of DiT.unpatchify: [B,C,H,W] -> [B, N, p*p*C] (same ordering)."""
    B, C, H, W = x.shape
    h, w = H // p, W // p
    x = x.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)          # B,h,w,p,p,C
    return x.reshape(B, h * w, p * p * C)


def gather_tokens(x, idx):
    """x:[B,N,D], idx:[B,k] -> [B,k,D]"""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def scatter_tokens(x, idx, val):
    """Replace rows of x:[B,N,D] at idx:[B,k] with val:[B,k,D] (out-of-place)."""
    return x.scatter(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]), val)


def topk_index(score, ratio, largest=True):
    """score:[B,N] -> hard-token indices [B,k], k = round(ratio*N) (>=1).
    largest=True keeps the highest-score tokens (score = risk/priority)."""
    B, N = score.shape
    k = max(1, int(round(ratio * N)))
    return torch.topk(score, k, dim=1, largest=largest).indices


# ---------------------------------------------------- forward pass primitives
def dit_embed(model, x, t, y):
    """Patch-embed + pos + conditioning. Returns (tok [B,N,D], c [B,D], hw)."""
    tok, hw = model.x_embed(x)
    tok = tok + model.pos
    c = model.t_embed(t) + model.y_embed(y, False)
    return tok, c, hw


def dit_forward_tokens(model, x, t, y):
    """Full dense forward that also returns the final hidden token states
    (input to FinalLayer). Used for draft-feature extraction (router input)
    and dense references. Returns (eps [B,C,H,W], hidden [B,N,D])."""
    tok, c, hw = dit_embed(model, x, t, y)
    for blk in model.blocks:
        tok = blk(tok, c)
    out = model.final(tok, c)
    return model.unpatchify(out, hw), tok


def dit_forward_prefix(model, x, t, y, m):
    """Run embed + first m blocks densely. Returns (tok, c, hw)."""
    tok, c, hw = dit_embed(model, x, t, y)
    for blk in model.blocks[:m]:
        tok = blk(tok, c)
    return tok, c, hw


# ----------------------------------------------------------- sparse suffixes
def _block_sparse_mlp(blk, x, c, idx):
    """Dense attention, MLP only on hard tokens idx:[B,k]."""
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = blk.ada(c).chunk(6, dim=1)
    h = modulate(blk.norm1(x), sh_msa, sc_msa)
    x = x + g_msa.unsqueeze(1) * blk.attn(h, h, h, need_weights=False)[0]
    xh = gather_tokens(x, idx)                                  # [B,k,D]
    h2 = modulate(blk.norm2(xh), sh_mlp, sc_mlp)
    xh = xh + g_mlp.unsqueeze(1) * blk.mlp(h2)
    return scatter_tokens(x, idx, xh)


def _block_sparse_attn(blk, x, c, idx):
    """Hard tokens as the only queries (all tokens are K/V context); MLP also
    hard-only. Easy tokens receive NO update in this block."""
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = blk.ada(c).chunk(6, dim=1)
    h = modulate(blk.norm1(x), sh_msa, sc_msa)                  # all tokens (K/V)
    q = gather_tokens(h, idx)                                   # [B,k,D]
    attn_out = blk.attn(q, h, h, need_weights=False)[0]         # [B,k,D]
    xh = gather_tokens(x, idx) + g_msa.unsqueeze(1) * attn_out
    h2 = modulate(blk.norm2(xh), sh_mlp, sc_mlp)
    xh = xh + g_mlp.unsqueeze(1) * blk.mlp(h2)
    return scatter_tokens(x, idx, xh)


def dit_forward_suffix_sparse(model, tok, c, hw, idx, m, mode="sparse_mlp",
                              refresh_every=0):
    """Run blocks m..L-1 sparsely, then FinalLayer on hard tokens only.
    refresh_every=s > 0 runs every s-th suffix block DENSELY (all tokens),
    bounding easy-token staleness: stale-but-valid context degrades with
    suffix depth, and a periodic dense refresh resets it (Stage-A finding:
    identity easy handling loses most of the budget advantage without this).
    Returns hard-token outputs [B, k, p*p*C] (caller scatters into a draft
    canvas)."""
    blocks = model.blocks[m:]
    if mode == "dense":
        for blk in blocks:
            tok = blk(tok, c)
    elif mode in ("sparse_mlp", "sparse_attn"):
        step_fn = _block_sparse_mlp if mode == "sparse_mlp" else _block_sparse_attn
        for i, blk in enumerate(blocks):
            if refresh_every and (i + 1) % refresh_every == 0:
                tok = blk(tok, c)                               # dense refresh
            else:
                tok = step_fn(blk, tok, c, idx)
    else:
        raise ValueError(f"unknown suffix mode {mode}")
    tok_h = gather_tokens(tok, idx)
    # FinalLayer on hard tokens only
    shift, scale = model.final.ada(c).chunk(2, dim=1)
    return model.final.lin(modulate(model.final.norm(tok_h), shift, scale))


def sparse_target_eps(model, x, t, y, idx, m, mode, eps_canvas,
                      refresh_every=0):
    """Full sparse target pass: dense prefix (blocks 0..m-1) + sparse suffix
    (blocks m..L-1) + hard-only FinalLayer, scattered into `eps_canvas`
    (typically the draft prediction, [B,C,H,W]). Returns mixed eps [B,C,H,W]."""
    tok, c, hw = dit_forward_prefix(model, x, t, y, m)
    out_h = dit_forward_suffix_sparse(model, tok, c, hw, idx, m, mode,
                                      refresh_every)
    canvas = patchify_img(eps_canvas, model.patch)              # [B,N,p*p*C]
    canvas = scatter_tokens(canvas, idx, out_h)
    return model.unpatchify(canvas, hw)


# ------------------------------------------------- temporal cache (Stage 13)
def dit_forward_dense_with_cache(model, x, t, y, m):
    """Dense forward that also records the INPUT hidden states of every suffix
    block (blocks m..L-1). These are depth-correct states reused for easy
    tokens at subsequent sparse steps (temporal staleness <= cache period,
    instead of the depth-mismatched frozen states that Stage A2 showed to
    fail). Returns (eps, cache list of length L-m, each [B,N,D])."""
    tok, hw = model.x_embed(x)
    tok = tok + model.pos
    c = model.t_embed(t) + model.y_embed(y, False)
    cache = []
    for j, blk in enumerate(model.blocks):
        if j >= m:
            cache.append(tok)
        tok = blk(tok, c)
    out = model.final(tok, c)
    return model.unpatchify(out, hw), cache


def dit_forward_suffix_cached(model, tok_m, c, idx, m, cache):
    """Sparse suffix where easy-token context comes from the anchor-step cache
    (depth-correct, time-stale) instead of being frozen at depth m.

    tok_m : fresh prefix output of THIS step [B,N,D] (block-m input)
    cache : list from dit_forward_dense_with_cache (anchor step)

    Per block j: context = fresh hard states scattered into cached easy states
    (block m uses this step's fresh states for everyone — they exist for free);
    hard tokens are the only queries and the only MLP rows. Returns hard-token
    FinalLayer outputs [B,k,p*p*C]."""
    x_hard = gather_tokens(tok_m, idx)
    for j, blk in enumerate(model.blocks[m:]):
        base = tok_m if j == 0 else cache[j]
        ctx = scatter_tokens(base, idx, x_hard)
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = blk.ada(c).chunk(6, dim=1)
        h = modulate(blk.norm1(ctx), sh_msa, sc_msa)
        q = gather_tokens(h, idx)
        x_hard = x_hard + g_msa.unsqueeze(1) * blk.attn(q, h, h,
                                                        need_weights=False)[0]
        h2 = modulate(blk.norm2(x_hard), sh_mlp, sc_mlp)
        x_hard = x_hard + g_mlp.unsqueeze(1) * blk.mlp(h2)
    shift, scale = model.final.ada(c).chunk(2, dim=1)
    return model.final.lin(modulate(model.final.norm(x_hard), shift, scale))


def sparse_target_eps_cached(model, x, t, y, idx, m, eps_canvas, cache):
    """Cache-context sparse target pass: dense prefix + cached sparse suffix,
    hard outputs scattered into eps_canvas (draft prediction)."""
    tok, c, hw = dit_forward_prefix(model, x, t, y, m)
    out_h = dit_forward_suffix_cached(model, tok, c, idx, m, cache)
    canvas = patchify_img(eps_canvas, model.patch)
    canvas = scatter_tokens(canvas, idx, out_h)
    return model.unpatchify(canvas, hw)


# ------------------------------------------------------------ FLOPs accounting
def dit_block_flops(N, D, k=None, mode="dense", mlp_ratio=4.0):
    """Multiply-accumulate count for one DiT block (per image).
    dense       : attn proj 4*N*D^2 + attn matmul 2*N^2*D + mlp 2*mlp_ratio*N*D^2
    sparse_mlp  : same attention, mlp on k tokens
    sparse_attn : q proj on k, kv+out proj mixed, matmul 2*k*N*D, mlp on k
    """
    mlp = 2 * mlp_ratio * D * D
    if mode == "dense":
        return N * 4 * D * D + 2 * N * N * D + N * mlp
    if mode == "sparse_mlp":
        return N * 4 * D * D + 2 * N * N * D + k * mlp
    if mode == "sparse_attn":
        # q proj: k*D^2 ; k,v proj: 2*N*D^2 ; out proj: k*D^2 ; scores+values: 2*k*N*D
        return (2 * k + 2 * N) * D * D + 2 * k * N * D + k * mlp
    raise ValueError(mode)


def dit_model_flops(model, mode="dense", m=None, k=None, refresh_every=0):
    """Total block+final MAC count per image for a (possibly sparse) forward."""
    N, D, L = model.num_tokens, model.pos.shape[-1], len(model.blocks)
    p, C = model.patch, model.out_ch
    fin_tokens = N if (mode == "dense" or k is None) else k
    fl = fin_tokens * D * (p * p * C)                    # FinalLayer linear
    if mode == "dense" or m is None:
        return L * dit_block_flops(N, D, mode="dense") + N * D * (p * p * C)
    n_suffix = L - m
    n_refresh = (n_suffix // refresh_every) if refresh_every else 0
    fl_prefix = m * dit_block_flops(N, D, mode="dense")
    fl_suffix = ((n_suffix - n_refresh) * dit_block_flops(N, D, k=k, mode=mode)
                 + n_refresh * dit_block_flops(N, D, mode="dense"))
    return fl_prefix + fl_suffix + fl


if __name__ == "__main__":
    # sanity: dense suffix path must reproduce DiT.forward exactly
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.dit import build_dit
    torch.manual_seed(0)
    m = build_dit("DiT-Nano", img_size=32, patch=4, num_classes=10,
                  class_dropout=0.0).eval()
    x = torch.randn(2, 3, 32, 32); t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        ref = m(x, t, y)
        eps2, hid = dit_forward_tokens(m, x, t, y)
        tok, c, hw = dit_forward_prefix(m, x, t, y, 2)
        idx = topk_index(torch.randn(2, m.num_tokens), 1.0)     # all tokens
        out = dit_forward_suffix_sparse(m, tok, c, hw,
                                        torch.sort(idx, dim=1).values, 2, "dense")
        canvas = patchify_img(torch.zeros_like(ref), m.patch)
        canvas = scatter_tokens(canvas, torch.sort(idx, dim=1).values, out)
        full = m.unpatchify(canvas, hw)
    print("dense-tokens max|d|", (ref - eps2).abs().max().item())
    print("prefix+dense-suffix max|d|", (ref - full).abs().max().item())
    print("sparse_mlp r=0.3 FLOPs ratio",
          dit_model_flops(m, "sparse_mlp", m=2, k=int(0.3 * m.num_tokens))
          / dit_model_flops(m, "dense"))
    print("sparse_attn r=0.3 FLOPs ratio",
          dit_model_flops(m, "sparse_attn", m=2, k=int(0.3 * m.num_tokens))
          / dit_model_flops(m, "dense"))
