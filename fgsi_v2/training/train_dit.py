#!/usr/bin/env python
"""
training/train_dit.py — train a class-conditional DiT on CIFAR-10 (eps-pred DDPM).

Trains the target (e.g. DiT-S) on the denoising objective; the draft (e.g. DiT-Ti)
can be trained the same way or distilled toward a frozen target's eps via
--distill_from, which maximizes draft/target agreement for the verifier PoC.

Examples:
    # target
    python -m training.train_dit --model DiT-S --out ckpt_dit/target.pt --steps 60000
    # draft, distilled from target
    python -m training.train_dit --model DiT-Ti --out ckpt_dit/draft.pt --steps 60000 \
        --distill_from ckpt_dit/target.pt --distill_weight 1.0
"""
import argparse
import os

import torch
import torch.nn.functional as F

from models.dit import build_dit, count_params
from training.scheduler import DDPMSchedule


def get_loader(args):
    from torchvision import datasets, transforms
    if args.dataset == "imagefolder":
        import glob
        from PIL import Image
        paths = []
        for e in ("png", "jpg", "jpeg"):
            paths += glob.glob(os.path.join(args.data_root, "**", f"*.{e}"), recursive=True)
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"no images under {args.data_root}")
        tf = transforms.Compose([transforms.RandomCrop(args.img_size, pad_if_needed=True),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])

        class ImgDS(torch.utils.data.Dataset):
            def __len__(self): return len(paths)
            def __getitem__(self, i):
                return tf(Image.open(paths[i]).convert("RGB")), 0  # single class

        ds = ImgDS()
    else:
        tf = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])
        ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                       num_workers=args.workers, drop_last=True,
                                       pin_memory=True)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm, alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def main(args):
    dev = torch.device(args.device)
    sch = DDPMSchedule(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                       beta_schedule="linear", device=dev)
    model = build_dit(args.model, img_size=args.img_size, patch=args.patch,
                      num_classes=args.num_classes, class_dropout=args.class_dropout).to(dev)
    ema = build_dit(args.model, img_size=args.img_size, patch=args.patch,
                    num_classes=args.num_classes, class_dropout=0.0).to(dev)
    ema.load_state_dict(model.state_dict()); ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"[dit] {args.model} params={count_params(model)/1e6:.2f}M tokens={model.num_tokens}")

    target = None
    if args.distill_from:
        target = build_dit(args.target_model, img_size=args.img_size, patch=args.patch,
                           num_classes=args.num_classes, class_dropout=0.0).to(dev)
        ck = torch.load(args.distill_from, map_location=dev)
        target.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        print(f"[dit] distilling from {args.distill_from} ({args.target_model})")

    loader = get_loader(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    step = 0
    model.train()
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            t = torch.randint(0, sch.num_train_timesteps, (x.shape[0],), device=dev)
            noise = torch.randn_like(x)
            x_t = sch.q_sample(x, noise, t)
            eps = model(x_t, t, y, train=True)
            loss = F.mse_loss(eps, noise)
            if target is not None:
                with torch.no_grad():
                    eps_t = target(x_t, t, y, train=False)
                loss = loss + args.distill_weight * F.mse_loss(eps, eps_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, args.ema_decay)
            step += 1
            if step % args.log_every == 0:
                print(f"[dit] step {step}/{args.steps} loss {loss.item():.4f}")
            if step % args.save_every == 0 or step >= args.steps:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                            "args": vars(args), "step": step}, args.out)
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "args": vars(args), "step": step}, args.out)
    print(f"[dit] saved {args.out}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="DiT-S")
    p.add_argument("--target_model", type=str, default="DiT-S",
                   help="arch of the --distill_from checkpoint")
    p.add_argument("--distill_from", type=str, default="")
    p.add_argument("--distill_weight", type=float, default=1.0)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "imagefolder"],
                   help="imagefolder = random-crop images under data_root (e.g. DIV2K 64x64)")
    p.add_argument("--out", type=str, default="./ckpt_dit/model.pt")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--class_dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())