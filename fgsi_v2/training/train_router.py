#!/usr/bin/env python
"""
training/train_router.py — Stage 7: train the draft-only hard-token router.

Loads the sharded teacher dataset from dit_router_dataset.py. Two objectives:

    rank (default) : MSE between router output and the per-(image, timestep)
                     rank-normalized disagreement label in [0,1]. Matches the
                     deployed decision exactly (per-image TopK within one step)
                     and is invariant to the timestep-dependent risk scale.
    bce            : asymmetric BCE at a fixed hard-token budget r_label:
                     y_i = 1 for the top-r_label tokens per (image, step);
                     lambda_hard > lambda_easy penalizes missed hard tokens
                     (false accepts) more.

Held-out shards are used for validation; reports Spearman correlation with the
oracle ranking and hard-token recall at r in {0.1, 0.3, 0.5, 0.7}.

Usage:
    python -m training.train_router \
        --data_dir /mnt/HDD_12TB/bam_ki/results/dit_in64/router_data \
        --out /mnt/HDD_12TB/bam_ki/ckpt_dit_in64/router_nano.pt \
        --loss rank --steps 20000
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.token_router import TokenRouter, rank_normalize


class ShardDataset(torch.utils.data.Dataset):
    """Loads all shards into RAM (fp16 features); one item = one (image, step)
    row with all N tokens."""
    def __init__(self, paths):
        hs, ss, tt, dd = [], [], [], []
        for pth in paths:
            sh = torch.load(pth, map_location="cpu")
            hs.append(sh["h"]); ss.append(sh["scal"])
            tt.append(sh["t"]); dd.append(sh["d_eps"])
        self.h = torch.cat(hs); self.scal = torch.cat(ss)
        self.t = torch.cat(tt); self.d = torch.cat(dd)

    def __len__(self):
        return self.h.shape[0]

    def __getitem__(self, i):
        return self.h[i], self.scal[i], self.t[i], self.d[i]


@torch.no_grad()
def evaluate(router, loader, dev, ratios=(0.1, 0.3, 0.5, 0.7)):
    router.eval()
    sp_sum, n = 0.0, 0
    rec = {r: 0.0 for r in ratios}
    for h, scal, t, d in loader:
        h, scal, t, d = h.to(dev).float(), scal.to(dev).float(), t.to(dev), d.to(dev)
        s = router(h, scal, t)                                  # [B,N]
        rr, dr = rank_normalize(s), rank_normalize(d)
        # Spearman = Pearson of ranks
        rrc = rr - rr.mean(1, keepdim=True); drc = dr - dr.mean(1, keepdim=True)
        sp = (rrc * drc).sum(1) / (rrc.norm(dim=1) * drc.norm(dim=1) + 1e-8)
        sp_sum += sp.sum().item(); n += s.shape[0]
        N = s.shape[1]
        for r in ratios:
            k = max(1, int(round(r * N)))
            pred = s.topk(k, dim=1).indices
            true = d.topk(k, dim=1).indices
            pm = torch.zeros_like(s, dtype=torch.bool).scatter_(1, pred, True)
            tm = torch.zeros_like(s, dtype=torch.bool).scatter_(1, true, True)
            rec[r] += ((pm & tm).sum(1).float() / k).sum().item()
    router.train()
    return sp_sum / n, {r: rec[r] / n for r in ratios}


def main(args):
    dev = torch.device(args.device)
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.pt")))
    if not shards:
        raise FileNotFoundError(f"no shards under {args.data_dir}")
    n_val = max(1, int(round(args.val_frac * len(shards))))
    tr_ds = ShardDataset(shards[:-n_val] if len(shards) > n_val else shards)
    va_ds = ShardDataset(shards[-n_val:])
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    print(f"[router] shards train={len(shards)-n_val} val={n_val} "
          f"rows train={len(tr_ds)} val={len(va_ds)} "
          f"d_hidden={meta['d_hidden_draft']} tokens={meta['n_tokens']}")

    tr = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    va = torch.utils.data.DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                                     num_workers=args.workers)

    router = TokenRouter(meta["d_hidden_draft"], t_dim=args.t_dim,
                         width=args.width).to(dev)
    opt = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[router] params {sum(p.numel() for p in router.parameters())/1e3:.1f}k "
          f"loss={args.loss}")

    step, best_sp = 0, -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    while step < args.steps:
        for h, scal, t, d in tr:
            h, scal = h.to(dev).float(), scal.to(dev).float()
            t, d = t.to(dev), d.to(dev)
            s = router(h, scal, t)                              # [B,N]
            if args.loss == "rank":
                loss = F.mse_loss(torch.sigmoid(s), rank_normalize(d))
            else:  # asymmetric bce at fixed budget
                N = d.shape[1]
                k = max(1, int(round(args.r_label * N)))
                yb = torch.zeros_like(d).scatter_(
                    1, d.topk(k, dim=1).indices, 1.0)
                w = yb * args.lambda_hard + (1 - yb) * args.lambda_easy
                loss = (F.binary_cross_entropy_with_logits(
                    s, yb, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[router] step {step}/{args.steps} loss {loss.item():.5f}")
            if step % args.eval_every == 0 or step >= args.steps:
                sp, rec = evaluate(router, va, dev)
                print(f"[router] step {step} val spearman {sp:.4f} "
                      + " ".join(f"rec@{r}={rec[r]:.3f}" for r in rec))
                if sp > best_sp:
                    best_sp = sp
                    torch.save(dict(router=router.state_dict(),
                                    d_hidden_draft=meta["d_hidden_draft"],
                                    t_dim=args.t_dim, width=args.width,
                                    val_spearman=sp, recall=rec,
                                    args=vars(args), step=step), args.out)
                    print(f"[router] saved {args.out} (spearman {sp:.4f})")
            if step >= args.steps:
                break
    print(f"[router] done, best val spearman {best_sp:.4f} -> {args.out}")


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--loss", type=str, default="rank", choices=["rank", "bce"])
    ap.add_argument("--r_label", type=float, default=0.3,
                    help="hard-token budget for bce labels")
    ap.add_argument("--lambda_hard", type=float, default=3.0)
    ap.add_argument("--lambda_easy", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--t_dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64,
                    help="rows per batch (each row = all tokens of one step)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())
