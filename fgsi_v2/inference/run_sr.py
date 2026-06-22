"""
End-to-end FreqSpec super-resolution inference.
HR image -> LR (bicubic x1/scale) -> x4-upscaler target-only vs FreqSpec-SR.

This is the AAAI-track SR instantiation. It reuses the *same* generalized
verifier (inference.speculative_general.fgsr_refine) as inpainting; only the
task setup differs: region = whole field, condition = low-res image, no
known-region blending, wavelet-only saliency.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.sr_target_wrapper import SRTargetWrapper
from models.draft import DraftEpsUNet
from models.wavelet import DWT2D, lwd_wavelet_saliency
from training.scheduler import DDPMSchedule
from inference.speculative_general import fgsr_refine, baseline_refine
from utils.metrics import psnr, ssim, hh_band_psnr


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(img).unsqueeze(0)


def save_rgb(t, path):
    t = (t.float().clamp(-1, 1) + 1) / 2
    Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(path)


def save_gray(t, path):
    Image.fromarray((t[0, 0].float().cpu().numpy() * 255).clip(0, 255).astype("uint8")).save(path)


def build_draft(args, latent_ch, num_train_timesteps, device):
    """SR draft: cond_ch=3 (LR RGB), no mask channel -> 7ch input."""
    draft_kwargs = {
        "latent_ch": latent_ch,
        "num_train_timesteps": num_train_timesteps,
        "cond_ch": 3,
        "use_mask": False,
    }
    if args.draft_ckpt and os.path.isfile(args.draft_ckpt):
        ck = torch.load(args.draft_ckpt, map_location=device)
        saved = ck.get("args", {})
        if "draft_base_ch" in saved:
            draft_kwargs["base_ch"] = saved["draft_base_ch"]
            draft_kwargs["ch_mult"] = tuple(saved["draft_ch_mult"])
            draft_kwargs["t_dim"] = saved["draft_t_dim"]
        draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
        if args.use_ema_draft and ck.get("ema_draft") is not None:
            draft.load_state_dict(ck["ema_draft"])
            print(f"[sr] loaded EMA draft from {args.draft_ckpt}")
        else:
            draft.load_state_dict(ck["draft"])
            print(f"[sr] loaded draft {args.draft_ckpt}")
    else:
        draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
        print("[sr] no draft ckpt -> random weights (dummy/smoke only)")
    print(f"[sr] draft params: {sum(p.numel() for p in draft.parameters())/1e6:.2f}M")
    return draft


def main(args):
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}.get(args.target_dtype, torch.float32)
    target = SRTargetWrapper(model_id=args.target_id, device=device, dtype=dtype,
                             default_noise_level=args.noise_level)
    if not target.available:
        print("[sr] WARNING: target dummy mode, results not meaningful")

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

    draft = build_draft(args, target.latent_ch, sch.num_train_timesteps, device)
    dwt = DWT2D("haar").to(device)

    hr_size = args.lr_size * args.scale
    hr = load_image(args.image, hr_size).to(device)        # ground-truth HR [-1,1]
    # low-res conditioning at *latent* spatial resolution (= lr_size)
    lr = F.interpolate(hr, size=(args.lr_size, args.lr_size),
                       mode="bicubic", align_corners=False).clamp(-1, 1)

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        cond_lr, nl_tensor = target.prepare_lr_cond(lr, noise_level=args.noise_level)
        Hl = Wl = args.lr_size
        region = torch.ones(1, 1, Hl, Wl, device=device)   # whole field verified

    # wavelet saliency visualization on the LR cond (HF regions = strict)
    sal = lwd_wavelet_saliency(cond_lr if cond_lr.shape[1] == 4 else
                               F.pad(cond_lr, (0, 0, 0, 0)),
                               dwt, target_size=(Hl, Wl)) \
        if cond_lr.shape[1] >= 3 else None

    z_init = torch.randn(1, target.latent_ch, Hl, Wl, device=device)

    if target.available:
        cond_emb, uncond_emb, use_cfg = target.get_text_embeddings(
            args.prompt, batch_size=1, guidance_scale=args.guidance_scale)
        print(f"[sr] CFG={'on' if use_cfg else 'off'} guidance={args.guidance_scale} "
              f"prompt='{args.prompt}' noise_level={args.noise_level}")
    else:
        cond_emb, uncond_emb = None, None

    extra = {"noise_level": nl_tensor}

    # ---- baseline (target-only) ----
    t0 = time.time()
    z_base, s_base = baseline_refine(
        target, z_init.clone(), cond_lr, region, sch,
        num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale,
        cond_emb=cond_emb, uncond_emb=uncond_emb,
        known_z=None, blend_known=False, target_extra=extra,
    )
    t_base = time.time() - t0

    # ---- FreqSpec-SR ----
    t0 = time.time()
    z_spec, s_spec = fgsr_refine(
        target, draft, z_init.clone(), cond_lr, region, sch,
        num_inference_steps=args.num_steps,
        K=args.K, patch_size=args.patch,
        t_spec_start_norm=args.t_spec_start, beta=args.beta,
        tol_low=args.tol_low, tol_high=args.tol_high,
        boundary_weight=0.0, mask_interior_weight=0.0,   # SR: wavelet-only saliency
        guidance_scale=args.guidance_scale,
        cond_emb=cond_emb, uncond_emb=uncond_emb,
        known_z=None, blend_known=False,
        blend_temperature=args.blend_temperature,
        x0_thr_strict=args.x0_thr_strict, x0_thr_loose=args.x0_thr_loose,
        x0_strict_center=args.x0_strict_center, x0_strict_width=args.x0_strict_width,
        drift_k_switch_threshold=args.drift_k_switch,
        k_switch_threshold=args.k_switch,
        dwt=dwt, verbose=args.verbose, target_extra=extra,
    )
    t_spec = time.time() - t0

    out_base = target.decode_latent(z_base)
    out_spec = target.decode_latent(z_spec)

    save_rgb(hr, os.path.join(args.out_dir, "hr_gt.png"))
    save_rgb(lr, os.path.join(args.out_dir, "lr_input.png"))
    save_rgb(out_base, os.path.join(args.out_dir, "out_baseline.png"))
    save_rgb(out_spec, os.path.join(args.out_dir, "out_freqspec.png"))

    if target.available:
        with torch.no_grad():
            print(f"\n[metrics vs HR ground-truth]")
            for tag, out in (("baseline", out_base), ("freqspec", out_spec)):
                print(f"  {tag:9s} PSNR={psnr(out, hr).item():.2f}  "
                      f"SSIM={ssim(out, hr).item():.4f}  "
                      f"HH-PSNR={hh_band_psnr(out, hr, dwt).item():.2f}")
            # trajectory divergence (FreqSpec vs target-only output)
            print(f"  div(freqspec, baseline) PSNR="
                  f"{psnr(out_spec, out_base).item():.2f}")

    print(f"\n[result] baseline (target-only): time={t_base:.2f}s "
          f"NFE_target={s_base['target_calls']}")
    print(f"[result] FreqSpec-SR: time={t_spec:.2f}s "
          f"NFE_target={s_spec['target_calls']} NFE_draft={s_spec['draft_calls']} "
          f"accept={s_spec['accept_rate']:.3f} "
          f"target_speedup={s_spec['target_speedup']:.2f}x "
          f"wall_speedup={t_base/max(t_spec,1e-6):.2f}x")
    print(f"\nsaved -> {args.out_dir}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="HR ground-truth image")
    p.add_argument("--draft_ckpt", type=str, default="")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-x4-upscaler")
    p.add_argument("--out_dir", type=str, default="./results_sr")
    p.add_argument("--lr_size", type=int, default=128,
                   help="Low-res / latent spatial size. HR = lr_size * scale.")
    p.add_argument("--scale", type=int, default=4, help="Upscale factor (x4 model).")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--noise_level", type=int, default=20,
                   help="x4-upscaler low-res noise level (class conditioning).")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="x4 upscaler typically uses low/no CFG.")
    p.add_argument("--target_dtype", type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--use_ema_draft", action="store_true")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--t_spec_start", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--tol_low", type=float, default=0.03)
    p.add_argument("--tol_high", type=float, default=0.30)
    # Combo-2 calibration defaults (mirrors the inpainting paper's Combo 2)
    p.add_argument("--blend_temperature", type=float, default=0.10)
    p.add_argument("--x0_thr_strict", type=float, default=0.02)
    p.add_argument("--x0_thr_loose", type=float, default=0.07)
    p.add_argument("--x0_strict_center", type=float, default=0.45)
    p.add_argument("--x0_strict_width", type=float, default=0.12)
    p.add_argument("--drift_k_switch", type=float, default=0.006)
    p.add_argument("--k_switch", type=float, default=0.60)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
