#!/usr/bin/env python
"""
utils/inpaint_masks.py — inpainting masks and token-level mask machinery for
the DiT-inpainting DACE experiments.

Pixel-level (matches the FreqSpec-Inpaint training distribution, scaled to
64x64): box masks with probability 0.4, irregular brush strokes otherwise.
mask = 1 inside the hole (region to reconstruct), 0 on known context.

Token-level (patch-grid) operators used by the sparse sampler:
    mask_to_tokens      : pixel mask -> per-token interior fraction [B,N]
    dilate_tokens       : morphological dilation on the token grid
    boundary_band       : dilate(M) - erode(M) on the token grid
    block_round         : round a hard-token index set UP to bxb token blocks
                          (structured sparsity; inpainting masks are contiguous
                          so this is nearly free in budget)
    combo_score         : s_i = a*M_i + b*B_i + g*F_i + d*D_i with per-image
                          rank normalization of the continuous signals

All token maps are [B, 1, h, w] or flattened [B, N] with N = h*w.
"""
import math

import torch
import torch.nn.functional as F


# ----------------------------------------------------------- pixel-level masks
def _box_mask(H, W, gen, min_frac=0.25, max_frac=0.5, device="cpu"):
    m = torch.zeros(1, H, W, device=device)
    bh = int(H * (min_frac + (max_frac - min_frac) * torch.rand((), generator=gen).item()))
    bw = int(W * (min_frac + (max_frac - min_frac) * torch.rand((), generator=gen).item()))
    top = torch.randint(0, H - bh + 1, (1,), generator=gen).item()
    left = torch.randint(0, W - bw + 1, (1,), generator=gen).item()
    m[:, top:top + bh, left:left + bw] = 1.0
    return m


def _brush_mask(H, W, gen, n_strokes=(1, 4), width=(3, 9), n_vertex=(4, 10),
                device="cpu"):
    """Irregular brush strokes (random walk polylines drawn with a disc)."""
    m = torch.zeros(H, W, device=device)
    ns = torch.randint(n_strokes[0], n_strokes[1] + 1, (1,), generator=gen).item()
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    for _ in range(ns):
        nv = torch.randint(n_vertex[0], n_vertex[1] + 1, (1,), generator=gen).item()
        w = torch.randint(width[0], width[1] + 1, (1,), generator=gen).item()
        x = torch.randint(0, W, (1,), generator=gen).item()
        y = torch.randint(0, H, (1,), generator=gen).item()
        ang = torch.rand((), generator=gen).item() * 2 * math.pi
        for _ in range(nv):
            ang += (torch.rand((), generator=gen).item() - 0.5) * 1.6
            ln = 4 + torch.rand((), generator=gen).item() * (min(H, W) / 3)
            nx = min(max(x + ln * math.cos(ang), 0), W - 1)
            ny = min(max(y + ln * math.sin(ang), 0), H - 1)
            steps = max(2, int(ln))
            for s in range(steps + 1):
                cx = x + (nx - x) * s / steps
                cy = y + (ny - y) * s / steps
                m = torch.maximum(m, ((xx - cx) ** 2 + (yy - cy) ** 2
                                      <= (w / 2) ** 2).float())
            x, y = nx, ny
    return m.unsqueeze(0)


def sample_masks(B, H, W, box_prob=0.4, seed=None, device="cpu"):
    """[B,1,H,W] float in {0,1}; 1 = hole. Deterministic per (seed, index)."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    out = []
    for _ in range(B):
        if torch.rand((), generator=gen).item() < box_prob:
            out.append(_box_mask(H, W, gen, device=device))
        else:
            out.append(_brush_mask(H, W, gen, device=device))
    return torch.stack(out, 0)


# ---------------------------------------------------------- token-level utils
def mask_to_tokens(mask_px, patch):
    """[B,1,H,W] pixel mask -> per-token interior fraction [B,1,h,w]."""
    return F.avg_pool2d(mask_px, patch, stride=patch)


def dilate_tokens(tok_map, k=1):
    """Binary/soft dilation with a (2k+1) square structuring element."""
    if k <= 0:
        return tok_map
    return F.max_pool2d(tok_map, 2 * k + 1, stride=1, padding=k)


def erode_tokens(tok_map, k=1):
    if k <= 0:
        return tok_map
    return 1.0 - F.max_pool2d(1.0 - tok_map, 2 * k + 1, stride=1, padding=k)


def boundary_band(tok_bin, k=1):
    """Morphological gradient on the TOKEN grid: dilate(k) - erode(k).
    review item 4: this k is the SELECTION boundary width
    (selection_boundary_k) and is kept separate from the EXECUTION mask
    dilation (execution_mask_dilate) so the two roles do not overlap."""
    return (dilate_tokens(tok_bin, k) - erode_tokens(tok_bin, k)).clamp(0, 1)


def pixel_boundary_ring(mask_px, ring=6):
    """Pixel-space boundary ring around the hole edge for boundary-LPIPS:
    dilate(mask, ring) - erode(mask, ring), in {0,1} [B,1,H,W]."""
    d = F.max_pool2d(mask_px, 2 * ring + 1, stride=1, padding=ring)
    e = 1.0 - F.max_pool2d(1.0 - mask_px, 2 * ring + 1, stride=1, padding=ring)
    return (d - e).clamp(0, 1)


def rank_normalize(score_flat):
    """Per-image rank in [0,1] (robust to per-timestep scale, matches the
    per-timestep normalization finding of the One Verifier paper)."""
    B, N = score_flat.shape
    rk = score_flat.argsort(dim=1).argsort(dim=1).float()
    return rk / max(1, N - 1)


def combo_score(M_tok=None, B_tok=None, F_tok=None, D_tok=None,
                a=1.0, b=1.0, g=0.5, d=0.5):
    """s_i = a*M_i + b*B_i + g*rank(F_i) + d*rank(D_i). Any component can be
    None. All inputs [B,N]; returns [B,N]."""
    s = None

    def _acc(s, w, x, ranked):
        if x is None or w == 0:
            return s
        x = rank_normalize(x) if ranked else x
        return w * x if s is None else s + w * x

    s = _acc(s, a, M_tok, False)
    s = _acc(s, b, B_tok, False)
    s = _acc(s, g, F_tok, True)
    s = _acc(s, d, D_tok, True)
    return s


def block_round_indices(idx, h, w, block=2):
    """Round a hard-token index set [B,k] UP to full bxb token blocks and
    return a padded rectangular index tensor [B,k'] (k' = max block-covered
    count over the batch; shorter rows are padded by repeating their last
    index, which is harmless for gather/scatter). Structured sparsity for
    GPU-friendly execution — inpainting masks are contiguous so the budget
    overhead is small.

    NOTE (review item 6): this is *block-structured selection*, not a
    block-sparse kernel. Execution still uses rectangular gather/scatter, so
    the padded k' is what runs; the true per-sample block-covered count is
    returned separately for MAC accounting. A real block-sparse speed-up
    needs a fused grouped kernel."""
    B, k = idx.shape
    dev = idx.device
    covered = torch.zeros(B, h * w, dtype=torch.bool, device=dev)
    covered.scatter_(1, idx, True)
    cov = covered.view(B, 1, h, w).float()
    cov = F.max_pool2d(cov, block, stride=block)
    cov = F.interpolate(cov, scale_factor=block, mode="nearest")
    cov = cov.view(B, h * w) > 0.5
    counts = cov.sum(1)                                   # true per-sample k
    kk = int(counts.max().item())
    out = torch.zeros(B, kk, dtype=torch.long, device=dev)
    for i in range(B):
        ci = torch.nonzero(cov[i], as_tuple=False).squeeze(1)
        out[i, :ci.numel()] = ci
        if ci.numel() < kk:
            out[i, ci.numel():] = ci[-1]
    return torch.sort(out, dim=1).values, counts


def exact_set_indices(eligible_bin):
    """review item 3 — EXACT per-sample budget. Given a per-sample binary
    eligibility map [B,N] (e.g. dilated hole tokens), return
        idx    : [B, kmax] padded index tensor (padding = repeats of a real
                 in-set index, harmless for gather/scatter — no extra token is
                 ever refreshed)
        counts : [B] true per-sample |set|  (used for MAC accounting)
    Every sample refreshes EXACTLY its own eligible set, so a small mask is
    not forced to borrow tokens from outside its hole (the bug in the old
    batch-max auto budget)."""
    B, N = eligible_bin.shape
    dev = eligible_bin.device
    e = eligible_bin > 0.5
    counts = e.sum(1).clamp(min=1)
    kmax = int(counts.max().item())
    out = torch.zeros(B, kmax, dtype=torch.long, device=dev)
    for i in range(B):
        ci = torch.nonzero(e[i], as_tuple=False).squeeze(1)
        if ci.numel() == 0:                              # empty mask guard
            ci = torch.tensor([0], device=dev)
        out[i, :ci.numel()] = ci
        if ci.numel() < kmax:
            out[i, ci.numel():] = ci[-1]
    return torch.sort(out, dim=1).values, counts


def topk_within(score, eligible_bin, ratio_of_set):
    """Fixed-ratio budget CONFINED to an eligible set, per sample. k_b =
    ceil(ratio_of_set * |set_b|); score outside the set is set to -inf so the
    top-k stays inside. Returns padded idx [B,kmax] + true counts [B]. Used
    for the within-mask ranking ablations (mask+boundary+freq+delta), where a
    ratio < 1 exposes whether the ranking inside the hole matters."""
    B, N = score.shape
    dev = score.device
    e = eligible_bin > 0.5
    s = score.masked_fill(~e, float("-inf"))
    set_sz = e.sum(1).clamp(min=1)
    counts = (ratio_of_set * set_sz.float()).ceil().long().clamp(min=1)
    kmax = int(counts.max().item())
    order = s.argsort(dim=1, descending=True)            # [B,N]
    out = order[:, :kmax].contiguous()
    # pad rows whose k_b < kmax by repeating their last valid pick
    for i in range(B):
        kb = int(counts[i].item())
        if kb < kmax:
            out[i, kb:] = out[i, kb - 1]
    return torch.sort(out, dim=1).values, counts


def known_latent(x0, eps0, sqrt_ac, sqrt_1mac, t):
    """Deterministic known-region latent at integer timestep t:
        z_known(t) = sqrt(abar_t) * x0 + sqrt(1-abar_t) * eps0
    with a FIXED per-image noise eps0 so trajectories are seed-comparable.
    t = -1 returns x0 itself."""
    if t < 0:
        return x0
    return sqrt_ac[t] * x0 + sqrt_1mac[t] * eps0


if __name__ == "__main__":
    m = sample_masks(4, 64, 64, seed=0)
    print("mask frac:", m.flatten(1).mean(1).tolist())
    tok = mask_to_tokens(m, 4)
    bnd = boundary_band((tok > 0.5).float(), 1)
    print("token map", tok.shape, "boundary tokens per img",
          bnd.flatten(1).sum(1).tolist())
    idx = torch.topk(tok.flatten(1), 20, dim=1).indices
    br, cnt = block_round_indices(idx, 16, 16, block=2)
    print("block-rounded k:", br.shape, "true counts", cnt.tolist())
    ex, exc = exact_set_indices((tok.flatten(1) > 0.5).float())
    print("exact mask idx:", ex.shape, "true counts", exc.tolist())
