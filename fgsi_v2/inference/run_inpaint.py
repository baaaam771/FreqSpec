"""
End-to-end FGSI inference.
이미지 + 마스크 -> baseline (target-only DDIM) vs FGSR 비교.
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

from models.target_wrapper import TargetWrapper
from models.draft import DraftEpsUNet
from models.wavelet import DWT2D, combined_saliency
from training.scheduler import DDPMSchedule
from inference.speculative import fgsr_inpaint, baseline_inpaint


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])(img).unsqueeze(0)


def load_mask(path, size):
    m = Image.open(path).convert("L")
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])(m)
    return (t > 0.5).float().unsqueeze(0)


def save_rgb(t, path):
    t = (t.clamp(-1, 1) + 1) / 2
    Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(path)


def save_gray(t, path):
    Image.fromarray((t[0, 0].cpu().numpy() * 255).clip(0, 255).astype("uint8")).save(path)


def main(args):
    device = torch.device(args.device)
    target = TargetWrapper(model_id=args.target_id, device=device)
    if not target.available:
        print("[inference] WARNING: target dummy mode, results not meaningful")

    # scheduler
    if target.available:
        sch = DDPMSchedule(
            num_train_timesteps=target.scheduler_ref.config.num_train_timesteps,
            beta_start=target.scheduler_ref.config.beta_start,
            beta_end=target.scheduler_ref.config.beta_end,
            beta_schedule=target.scheduler_ref.config.beta_schedule,
            device=device,
        )
    else:
        sch = DDPMSchedule(device=device)

    # draft - architecture는 checkpoint의 저장된 args에서 자동으로 읽어옴
    draft_kwargs = {
        "latent_ch": target.latent_ch,
        "num_train_timesteps": sch.num_train_timesteps,
    }
    if args.draft_ckpt and os.path.isfile(args.draft_ckpt):
        ck = torch.load(args.draft_ckpt, map_location=device)
        # checkpoint에 학습 args가 있으면 그걸 따라 architecture 재구성
        saved_args = ck.get("args", {})
        if "draft_base_ch" in saved_args:
            draft_kwargs["base_ch"] = saved_args["draft_base_ch"]
            draft_kwargs["ch_mult"] = tuple(saved_args["draft_ch_mult"])
            draft_kwargs["t_dim"] = saved_args["draft_t_dim"]
        draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
        # EMA가 있으면 우선 사용
        if args.use_ema_draft and "ema_draft" in ck and ck["ema_draft"] is not None:
            draft.load_state_dict(ck["ema_draft"])
            print(f"[inference] loaded EMA draft from {args.draft_ckpt}")
        else:
            draft.load_state_dict(ck["draft"])
            print(f"[inference] loaded draft {args.draft_ckpt}")
        print(f"[inference] draft params: {sum(p.numel() for p in draft.parameters())/1e6:.2f}M")
    else:
        draft = DraftEpsUNet(**draft_kwargs).to(device).eval()
        print("[inference] no draft ckpt -> random weights")

    dwt = DWT2D("haar").to(device)

    # inputs
    img = load_image(args.image, args.image_size).to(device)
    if args.mask:
        mask_pix = load_mask(args.mask, args.image_size).to(device)
    else:
        mask_pix = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
        s = args.image_size // 4
        c = args.image_size // 2
        mask_pix[:, :, c - s:c + s, c - s:c + s] = 1.0

    os.makedirs(args.out_dir, exist_ok=True)

    masked_pix = img * (1 - mask_pix)
    with torch.no_grad():
        z0 = target.encode_image(img) if target.available else img
        Hl, Wl = z0.shape[-2:]
        cond_z = target.encode_image(masked_pix) if target.available else masked_pix
        mask_z = target.downsample_mask(mask_pix) if target.available else \
            F.interpolate(mask_pix, size=(Hl, Wl), mode="nearest")

    # saliency 시각화
    sal = combined_saliency(z0, mask_z, dwt,
                            boundary_weight=args.boundary_weight,
                            target_size=(args.image_size, args.image_size))
    save_gray(sal, os.path.join(args.out_dir, "saliency.png"))

    # SD-Inpainting 표준 방식: fully noisy z_init from scratch
    # (마스크 외부 보존은 매 step의 blending으로 처리)
    z_init = torch.randn_like(z0)

    # Text embeddings (for CFG)
    if target.available:
        cond_emb, uncond_emb, use_cfg = target.get_text_embeddings(
            args.prompt, batch_size=z0.shape[0], guidance_scale=args.guidance_scale
        )
        print(f"[inference] CFG={'on' if use_cfg else 'off'} guidance={args.guidance_scale} prompt='{args.prompt}'")
    else:
        cond_emb, uncond_emb = None, None

    # ---- baseline ----
    t0 = time.time()
    z_base, s_base = baseline_inpaint(
        target, z_init.clone(), cond_z, mask_z, sch,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        cond_emb=cond_emb, uncond_emb=uncond_emb,
        known_z=z0, blend_known=True,
    )
    t_base = time.time() - t0

    # ---- FGSR ----
    t0 = time.time()
    z_spec, s_spec = fgsr_inpaint(
        target, draft, z_init.clone(), cond_z, mask_z, sch,
        num_inference_steps=args.num_steps,
        K=args.K, patch_size=args.patch,
        t_spec_start_norm=args.t_spec_start,
        beta=args.beta,
        tol_low=args.tol_low, tol_high=args.tol_high,
        boundary_weight=args.boundary_weight,
        dwt=dwt, verbose=args.verbose,
        guidance_scale=args.guidance_scale,
        cond_emb=cond_emb, uncond_emb=uncond_emb,
        known_z=z0, blend_known=True,
    )
    t_spec = time.time() - t0

    out_base = target.decode_latent(z_base) if target.available else z_base
    out_spec = target.decode_latent(z_spec) if target.available else z_spec
    out_base = img * (1 - mask_pix) + out_base * mask_pix
    out_spec = img * (1 - mask_pix) + out_spec * mask_pix

    save_rgb(img, os.path.join(args.out_dir, "input.png"))
    save_rgb(masked_pix, os.path.join(args.out_dir, "masked.png"))
    save_rgb(out_base, os.path.join(args.out_dir, "out_baseline.png"))
    save_rgb(out_spec, os.path.join(args.out_dir, "out_fgsr.png"))
    save_gray(mask_pix, os.path.join(args.out_dir, "mask.png"))

    print(f"\n[result] baseline (target-only DDIM):")
    print(f"  time={t_base:.2f}s  NFE_target={s_base['target_calls']}")
    print(f"[result] FGSR:")
    print(f"  time={t_spec:.2f}s  NFE_target={s_spec['target_calls']}  "
          f"NFE_draft={s_spec['draft_calls']}  "
          f"accept_rate={s_spec['accept_rate']:.3f}  "
          f"target_speedup={s_spec['target_speedup']:.2f}x")
    print(f"\nsaved -> {args.out_dir}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--mask", type=str, default="")
    p.add_argument("--draft_ckpt", type=str, default="")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-2-inpainting")
    p.add_argument("--out_dir", type=str, default="./results")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--prompt", type=str, default="",
                   help="Text prompt for CFG. Empty = unconditional.")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="CFG guidance scale. 1.0 = no CFG (faster, lower quality).")
    p.add_argument("--use_ema_draft", action="store_true",
                   help="Use EMA draft weights if available in checkpoint.")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--t_spec_start", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--tol_low", type=float, default=0.05)
    p.add_argument("--tol_high", type=float, default=0.5)
    p.add_argument("--boundary_weight", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())