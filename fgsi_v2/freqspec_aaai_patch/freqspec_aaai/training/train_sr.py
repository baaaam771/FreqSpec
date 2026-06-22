"""
FreqSpec-SR draft training.

Trains a compact draft as a local surrogate of the frozen x4-upscaler target.
Parallel to training/train.py but SR-specific:
  - region = whole field (no inpaint mask sampling)
  - conditioning = low-res image (noised, x4-upscaler convention)
  - region-aware loss reduces to: target-distillation on low-frequency (smooth)
    regions, ground-truth supervision on high-frequency (wavelet) regions.
    This is exactly the FreqSpec training/inference alignment, instantiated for
    super-resolution where "hard" == high-frequency detail.
"""
import os
import sys
import copy
import argparse
import random as _rnd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.sr_target_wrapper import SRTargetWrapper
from models.draft import DraftEpsUNet
from training.scheduler import DDPMSchedule
from training.losses import DraftLoss
from training.train import ImageDataset  # reuse recursive image dataset


def train(args):
    device = torch.device(args.device)
    print(f"[train-sr] device={device}")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}.get(args.target_dtype, torch.float32)
    target = SRTargetWrapper(model_id=args.target_id, device=device, dtype=dtype,
                             default_noise_level=args.noise_level)
    print(f"[train-sr] target available: {target.available}")

    if target.available:
        sch = DDPMSchedule(
            num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
            beta_start=target.scheduler_ref.config.beta_start,
            beta_end=target.scheduler_ref.config.beta_end,
            beta_schedule=getattr(target.scheduler_ref.config, "beta_schedule",
                                  "scaled_linear"),
            device=device,
        )
    else:
        sch = DDPMSchedule(device=device)

    # SR draft: cond_ch=3 (LR RGB), no mask channel -> 7ch input
    draft = DraftEpsUNet(
        latent_ch=target.latent_ch,
        base_ch=args.draft_base_ch,
        ch_mult=tuple(args.draft_ch_mult),
        t_dim=args.draft_t_dim,
        num_train_timesteps=sch.num_train_timesteps,
        cond_ch=3, use_mask=False,
    ).to(device)
    print(f"[train-sr] draft params: {sum(p.numel() for p in draft.parameters())/1e6:.2f}M")

    optim = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=1e-4)
    # region-aware loss; boundary off so the hard-region split is pure wavelet HF
    criterion = DraftLoss(
        boundary_weight=0.0,
        ell=args.ell,
        alpha_distill=args.alpha_distill,
        gamma_main=args.gamma_main,
        lambda_uniform=args.lambda_uniform,
        device=device,
        mask_signal="wavelet",
    )

    hr_size = args.lr_size * args.scale
    if args.data_root and os.path.isdir(args.data_root):
        ds = ImageDataset(args.data_root, image_size=hr_size,
                          default_prompt=args.default_prompt, return_prompt=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
        use_dummy = False
    else:
        print("[train-sr] no data_root -> dummy random tensors")
        loader, use_dummy = None, True

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    draft.train()

    if args.use_ema:
        ema_draft = copy.deepcopy(draft).eval()
        for p in ema_draft.parameters():
            p.requires_grad_(False)
    else:
        ema_draft = None

    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        draft.load_state_dict(ck["draft"])
        if ema_draft is not None and "ema_draft" in ck:
            ema_draft.load_state_dict(ck["ema_draft"])
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        step = ck.get("step", 0)
        print(f"[train-sr] resumed from {args.resume} at step {step}")

    if target.available:
        with torch.no_grad():
            uncond_full = target._encode_prompt([""] * args.batch_size)
    else:
        uncond_full = None

    for epoch in range(args.epochs):
        if use_dummy:
            iterator = (torch.randn(args.batch_size, 3, hr_size, hr_size)
                        for _ in range(args.steps_per_epoch))
        else:
            iterator = iter(loader)

        for batch in iterator:
            img = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            B = img.shape[0]

            with torch.no_grad():
                z0 = target.encode_image(img)  # wrapper handles dummy fallback (4ch)
                Hl, Wl = z0.shape[-2:]
                # LR conditioning at latent spatial resolution
                lr = F.interpolate(img, size=(Hl, Wl), mode="bicubic",
                                   align_corners=False).clamp(-1, 1)
                cond_lr, nl = target.prepare_lr_cond(lr, noise_level=args.noise_level)

                region = torch.ones(B, 1, Hl, Wl, device=device)
                eps_gt = torch.randn_like(z0)
                t = sch.sample_timesteps(B)
                z_t = sch.q_sample(z0, eps_gt, t)

                uncond = uncond_full[:B] if uncond_full is not None else None
                eps_target = target.predict_eps(
                    z_t, t, cond_lr, region,
                    cond_emb=None, uncond_emb=uncond,
                    guidance_scale=1.0, noise_level=nl,
                )

            eps_draft = draft(z_t, t, cond_lr, region)
            t_norm = sch.t_to_normalized(t)
            loss, logs, _, _ = criterion(eps_draft, eps_target, eps_gt, z0, region, t_norm)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optim.step()

            if ema_draft is not None:
                with torch.no_grad():
                    for pe, p in zip(ema_draft.parameters(), draft.parameters()):
                        pe.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

            if step % args.log_interval == 0:
                print(f"step{step} | loss={loss.item():.4f} "
                      f"l_dist={logs['l_distill']:.4f} l_main={logs['l_main']:.4f} "
                      f"l_unif={logs['l_uniform']:.4f} M_t={logs['M_t_active']:.3f}")

            if step > 0 and step % args.save_interval == 0:
                ck = {"draft": draft.state_dict(), "optim": optim.state_dict(),
                      "step": step, "args": vars(args)}
                if ema_draft is not None:
                    ck["ema_draft"] = ema_draft.state_dict()
                torch.save(ck, os.path.join(args.out_dir, f"draft_sr_step{step:07d}.pt"))
                torch.save(ck, os.path.join(args.out_dir, "draft_sr_latest.pt"))
                print(f"[train-sr] saved draft_sr_step{step:07d}.pt")

            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    ck = {"draft": draft.state_dict(), "optim": optim.state_dict(),
          "step": step, "args": vars(args)}
    if ema_draft is not None:
        ck["ema_draft"] = ema_draft.state_dict()
    torch.save(ck, os.path.join(args.out_dir, "draft_sr_final.pt"))
    torch.save(ck, os.path.join(args.out_dir, "draft_sr_latest.pt"))
    print(f"[train-sr] done. final saved to {args.out_dir}/draft_sr_final.pt")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./ckpt_sr")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-x4-upscaler")
    p.add_argument("--target_dtype", type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--lr_size", type=int, default=128)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--noise_level", type=int, default=20)
    p.add_argument("--default_prompt", type=str, default="")
    # draft architecture (82M default to match the inpainting draft)
    p.add_argument("--draft_base_ch", type=int, default=128)
    p.add_argument("--draft_ch_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    p.add_argument("--draft_t_dim", type=int, default=512)
    # loss
    p.add_argument("--alpha_distill", type=float, default=0.5)
    p.add_argument("--gamma_main", type=float, default=2.0)
    p.add_argument("--lambda_uniform", type=float, default=1.0)
    p.add_argument("--ell", type=float, default=0.3)
    # optim / schedule
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=400000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
