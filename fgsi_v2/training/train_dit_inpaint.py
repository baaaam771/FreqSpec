#!/usr/bin/env python
"""
training/train_dit_inpaint.py — train a mask-conditioned class-conditional
DiT inpainter on ImageNet-64 (eps-pred DDPM, pixel space).

Stage 1 of the DiT-inpainting DACE plan: before any sparse execution, a dense
DiT inpainting target must exist and be quality-gated. The draft (DiT-Nano-Inp)
is trained the same way with an optional region-aware distillation objective
adapted from FreqSpec-Inpaint Eq. 14:

    L = alpha_d * (1 - M_px) * ||eps - eps_tgt||^2      (easy: distill)
      + gamma_m *      M_px  * ||eps - noise||^2        (hole: GT noise)
      + lambda_u *            ||eps - noise||^2         (uniform safety)

where M_px is the pixel hole mask (the inpainting analog of the LWD
time-dependent hard mask; the hole IS the hard region here).

Examples (server, tmux):
    # target
    python -m training.train_dit_inpaint --model DiT-S-Inp \
        --data_root /mnt/HDD_12TB/bam_ki/imagenet64/train --dataset imagenet \
        --out /mnt/HDD_12TB/bam_ki/ckpt_dit_inp/target.pt --steps 300000 \
        --batch 256 --workers 0
    # draft, region-aware distilled
    python -m training.train_dit_inpaint --model DiT-Nano-Inp \
        --data_root /mnt/HDD_12TB/bam_ki/imagenet64/train --dataset imagenet \
        --out /mnt/HDD_12TB/bam_ki/ckpt_dit_inp/draft_nano.pt --steps 300000 \
        --distill_from /mnt/HDD_12TB/bam_ki/ckpt_dit_inp/target.pt \
        --target_model DiT-S-Inp --batch 256 --workers 0
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit_inpaint import build_dit_inpaint, count_params, DiTInpaint
from training.scheduler import DDPMSchedule
from utils.inpaint_masks import sample_masks
from training.train_dit import ema_update  # reuse EMA helper


def _worker_init(_wid):
    """서로 다른 worker가 다른 seed를 쓰도록."""
    import random
    import numpy as np
    s = torch.initial_seed() % (2 ** 31)
    np.random.seed(s)
    random.seed(s)


def get_loader_inpaint(args):
    """Inpainting 학습 전용 loader. train_dit.get_loader와 달리
    persistent_workers/prefetch_factor를 켜서 HDD의 작은 PNG random I/O
    병목에서도 GPU가 굶지 않게 한다 (train_dit.py는 공용이라 미변경)."""
    from torchvision import datasets, transforms
    if args.dataset in ("imagenet", "imagefolder"):
        tf = transforms.Compose([transforms.Resize(args.img_size),
                                 transforms.CenterCrop(args.img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])
        ds = datasets.ImageFolder(args.data_root, transform=tf)
        print(f"[dit-inp] ImageFolder: {len(ds)} images, "
              f"{len(ds.classes)} classes", flush=True)
    else:
        tf = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])
        ds = datasets.CIFAR10(args.data_root, train=True, download=True,
                              transform=tf)
    kw = dict(batch_size=args.batch, shuffle=True, drop_last=True,
              num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        # persistent_workers: epoch마다 worker 재생성 안 함 (forkserver 비용 제거)
        # prefetch_factor: worker당 미리 읽어둘 배치 수 → GPU 연산 중 I/O 겹침
        kw.update(persistent_workers=True,
                  prefetch_factor=args.prefetch_factor,
                  worker_init_fn=_worker_init)
    return torch.utils.data.DataLoader(ds, **kw)


def atomic_save(obj, path):
    """Write to a temp file then rename, so a crash mid-write never corrupts
    the checkpoint (tmux 세션이 죽어도 마지막 저장본은 항상 온전)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_ckpt(model, ema, opt, step, args):
    return {"model": model.state_dict(), "ema": ema.state_dict(),
            "opt": opt.state_dict(), "step": step, "args": vars(args),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None)}


def try_resume(model, ema, opt, args, dev):
    """Resume from --resume if given, else auto-resume from --out (or its
    .last.pt) when it exists and --fresh was not passed. Returns start step."""
    cand = []
    if args.resume:
        cand.append(args.resume)
    if not args.fresh:
        cand += [args.out, args.out + ".last.pt"]
    for path in cand:
        if path and os.path.exists(path):
            ck = torch.load(path, map_location=dev)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
            if "opt" in ck:
                try:
                    opt.load_state_dict(ck["opt"])
                except Exception as e:
                    print(f"[dit-inp] opt state skipped ({e})")
            if ck.get("torch_rng") is not None:
                torch.set_rng_state(ck["torch_rng"].to("cpu"))
            if ck.get("cuda_rng") is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(ck["cuda_rng"])
                except Exception:
                    pass
            step = int(ck.get("step", 0))
            print(f"[dit-inp] RESUME from {path} at step {step}/{args.steps}")
            return step
    print("[dit-inp] no checkpoint to resume; starting fresh")
    return 0


def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    model = build_dit_inpaint(args.model, img_size=args.img_size,
                              patch=args.patch, num_classes=args.num_classes,
                              class_dropout=args.class_dropout).to(dev)
    ema = build_dit_inpaint(args.model, img_size=args.img_size,
                            patch=args.patch, num_classes=args.num_classes,
                            class_dropout=0.0).to(dev)
    ema.load_state_dict(model.state_dict()); ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"[dit-inp] {args.model} params={count_params(model)/1e6:.2f}M "
          f"tokens={model.num_tokens}", flush=True)

    target = None
    if args.distill_from:
        target = build_dit_inpaint(args.target_model, img_size=args.img_size,
                                   patch=args.patch,
                                   num_classes=args.num_classes,
                                   class_dropout=0.0).to(dev)
        ck = torch.load(args.distill_from, map_location=dev)
        target.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        print(f"[dit-inp] region-aware distill from {args.distill_from}", flush=True)

    loader = get_loader_inpaint(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    step = try_resume(model, ema, opt, args, dev)
    if step >= args.steps:
        print(f"[dit-inp] already at/after target steps ({step}>={args.steps}); "
              f"nothing to do. Use --fresh or raise --steps to continue.")
        return
    model.train()
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            B = x.shape[0]
            mask = sample_masks(B, args.img_size, args.img_size,
                                box_prob=args.box_prob).to(dev)
            x_masked = x * (1.0 - mask)
            t = torch.randint(0, sch.num_train_timesteps, (B,), device=dev)
            noise = torch.randn_like(x)
            z_t = sch.q_sample(x, noise, t)
            x_in = DiTInpaint.pack(z_t, x_masked, mask)
            eps = model(x_in, t, y, train=True)

            if target is None:
                w = 1.0 + args.hole_weight * mask
                loss = (w * (eps - noise) ** 2).mean()
            else:
                with torch.no_grad():
                    eps_t = target(x_in, t, y, train=False)
                e_dist = ((eps - eps_t) ** 2)
                e_gt = ((eps - noise) ** 2)
                loss = (args.alpha_d * ((1 - mask) * e_dist).mean()
                        + args.gamma_m * (mask * e_gt).mean()
                        + args.lambda_u * e_gt.mean())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, args.ema_decay)
            step += 1
            if step % args.log_every == 0:
                print(f"[dit-inp] step {step}/{args.steps} loss {loss.item():.4f}", flush=True)
            if step % args.save_every == 0 or step >= args.steps:
                ck = make_ckpt(model, ema, opt, step, args)
                atomic_save(ck, args.out)                 # canonical
                atomic_save(ck, args.out + ".last.pt")    # resume anchor
            if args.milestone_every and step % args.milestone_every == 0:
                atomic_save(make_ckpt(model, ema, opt, step, args),
                            f"{args.out}.step{step}.pt")
            if step >= args.steps:
                break
    atomic_save(make_ckpt(model, ema, opt, step, args), args.out)
    atomic_save(make_ckpt(model, ema, opt, step, args), args.out + ".last.pt")
    print(f"[dit-inp] saved {args.out} (step {step})", flush=True)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="DiT-S-Inp")
    p.add_argument("--target_model", type=str, default="DiT-S-Inp",
                   help="arch of the --distill_from checkpoint")
    p.add_argument("--distill_from", type=str, default="")
    p.add_argument("--alpha_d", type=float, default=0.5,
                   help="easy-region (known) distillation weight")
    p.add_argument("--gamma_m", type=float, default=2.0,
                   help="hole-region GT-noise weight")
    p.add_argument("--lambda_u", type=float, default=1.0,
                   help="uniform safety weight")
    p.add_argument("--hole_weight", type=float, default=0.0,
                   help="target training: extra loss weight inside the hole "
                        "(0 = plain uniform MSE)")
    p.add_argument("--box_prob", type=float, default=0.4,
                   help="probability of a box mask (else brush), 0.4 matches "
                        "the FreqSpec training distribution")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--dataset", type=str, default="imagenet",
                   choices=["cifar10", "imagefolder", "imagenet"])
    p.add_argument("--out", type=str, default="./ckpt_dit_inp/model.pt")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--class_dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--workers", type=int, default=8,
                   help="dataloader worker 수. HDD의 작은 PNG random I/O "
                        "병목에는 4~8 권장. Python 3.14 forkserver에서 pickle "
                        "에러가 나면 낮추거나 0. 0이면 메인 스레드가 읽어 "
                        "GPU가 굶을 수 있음")
    p.add_argument("--prefetch_factor", type=int, default=6,
                   help="worker당 미리 읽어둘 배치 수 (workers>0일 때만)")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--milestone_every", type=int, default=50000,
                   help="also keep a permanent snapshot out.stepN.pt every N "
                        "steps (0 disables); out.pt / out.pt.last.pt are "
                        "always overwritten for resume")
    p.add_argument("--resume", type=str, default="",
                   help="explicit checkpoint to resume from; default is "
                        "auto-resume from --out / --out.last.pt if present")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing checkpoint and start from step 0")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
