"""
Draft 모델 학습 loop.

핵심:
- Target은 frozen pretrained (SD-Inpainting)
- Draft만 학습
- 각 step:
    1. image -> VAE encode -> z0
    2. random mask, noise, t
    3. z_t = √α̅_t z0 + √(1-α̅_t) eps
    4. target_eps = target.predict_eps(z_t, t, cond, mask)   [no grad]
    5. draft_eps  = draft (z_t, t, cond, mask)
    6. loss = DraftLoss(draft_eps, target_eps, gt_eps, ...)
    7. backward & update draft
"""
import os
import sys
import math
import random
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.target_wrapper import TargetWrapper
from models.draft import DraftEpsUNet
from training.scheduler import DDPMSchedule
from training.losses import DraftLoss


def random_inpaint_mask(B, H, W, device, p_box=0.4):
    """Box + irregular strokes."""
    masks = []
    for _ in range(B):
        m = torch.zeros(1, H, W, device=device)
        if random.random() < p_box:
            bh = random.randint(H // 8, H // 2)
            bw = random.randint(W // 8, W // 2)
            y = random.randint(0, H - bh); x = random.randint(0, W - bw)
            m[:, y:y + bh, x:x + bw] = 1.0
        else:
            for _ in range(random.randint(3, 8)):
                y = random.randint(0, H - 1); x = random.randint(0, W - 1)
                length = random.randint(H // 8, H // 3)
                angle = random.uniform(0, 2 * math.pi)
                thick = random.randint(3, max(4, H // 16))
                for s in range(length):
                    yy = int(y + s * math.sin(angle))
                    xx = int(x + s * math.cos(angle))
                    if 0 <= yy < H and 0 <= xx < W:
                        m[:, max(0, yy - thick):min(H, yy + thick),
                             max(0, xx - thick):min(W, xx + thick)] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0)


class ImageDataset(Dataset):
    """
    재귀 dataset: root 아래의 모든 .jpg/.jpeg/.png/.webp를 평평하게 모은다.
    Places2 같은 다층 폴더 구조에도 그대로 대응.
    Inpainting에는 class label이 필요 없으므로 ImageFolder 대신 직접 구현.

    NOTE: __init__에서 module 객체를 저장하지 않는다 (pickle 불가).
    Python 3.14+ forkserver 방식에서 multiprocessing dataloader 사용 가능하도록.
    """
    def __init__(self, root, image_size=512):
        self.root = root
        exts = (".jpg", ".jpeg", ".png", ".webp",
                ".JPG", ".JPEG", ".PNG", ".WEBP")
        self.paths = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(exts):
                    self.paths.append(os.path.join(dirpath, fn))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")
        print(f"[ImageDataset] found {len(self.paths)} images under {root}")
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # PIL을 함수 안에서 import하여 module 객체를 멤버로 저장하지 않음
        from PIL import Image as PILImage
        try:
            img = PILImage.open(self.paths[i]).convert("RGB")
            return self.tf(img)
        except Exception:
            # 깨진 파일 만나면 다음 인덱스 시도
            return self.__getitem__((i + 1) % len(self.paths))


def train(args):
    device = torch.device(args.device)
    print(f"[train] device={device}")

    # ---- target (frozen) ----
    target_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                    "fp32": torch.float32}.get(args.target_dtype, torch.float32)
    target = TargetWrapper(model_id=args.target_id, device=device,
                           dtype=target_dtype)
    print(f"[train] target available: {target.available} "
          f"(is_sdxl={target.is_sdxl if hasattr(target, 'is_sdxl') else False}, "
          f"dtype={args.target_dtype})")

    # ---- scheduler (target과 호환) ----
    if target.available:
        # SD scheduler config 그대로
        nts = target.scheduler_ref.config.num_train_timesteps
        bs = target.scheduler_ref.config.beta_start
        be = target.scheduler_ref.config.beta_end
        bsch = target.scheduler_ref.config.beta_schedule
        sch = DDPMSchedule(num_train_timesteps=nts, beta_start=bs, beta_end=be,
                           beta_schedule=bsch, device=device)
    else:
        sch = DDPMSchedule(device=device)
    print(f"[train] scheduler T={sch.num_train_timesteps}, β_schedule")

    # ---- draft ----
    draft = DraftEpsUNet(
        latent_ch=target.latent_ch,
        base_ch=args.draft_base_ch,
        ch_mult=tuple(args.draft_ch_mult),
        t_dim=args.draft_t_dim,
        num_train_timesteps=sch.num_train_timesteps,
    ).to(device)
    print(f"[train] draft params: {sum(p.numel() for p in draft.parameters()) / 1e6:.2f}M")

    optim = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = DraftLoss(
        boundary_weight=args.boundary_weight,
        boundary_kernel=args.boundary_kernel,
        ell=args.ell,
        alpha_distill=args.alpha_distill,
        gamma_main=args.gamma_main,
        lambda_uniform=args.lambda_uniform,
        device=device,
    )

    # ---- data ----
    if args.data_root and os.path.isdir(args.data_root):
        ds = ImageDataset(args.data_root, image_size=args.image_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
        use_dummy = False
        print(f"[train] dataset size={len(ds)}")
    else:
        print("[train] no data_root -> dummy random tensors")
        loader = None
        use_dummy = True

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    draft.train()

    # ---- EMA (Exponential Moving Average) for stable draft ----
    if args.use_ema:
        import copy
        ema_draft = copy.deepcopy(draft).eval()
        for p in ema_draft.parameters():
            p.requires_grad_(False)
        ema_decay = args.ema_decay
    else:
        ema_draft = None

    # ---- Resume from checkpoint ----
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        draft.load_state_dict(ck["draft"])
        if ema_draft is not None and "ema_draft" in ck:
            ema_draft.load_state_dict(ck["ema_draft"])
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        if "step" in ck:
            step = ck["step"]
        print(f"[train] resumed from {args.resume} at step {step}")

    # ---- CFG embeddings (pre-compute once for training) ----
    # Note: target._encode_prompt returns Tensor for SD2, tuple (hidden, pooled) for SDXL.
    # We keep the raw return so predict_eps can handle both formats.
    if args.train_with_cfg and target.available:
        with torch.no_grad():
            train_uncond_emb = target._encode_prompt([""] * args.batch_size)
            train_cond_emb = target._encode_prompt([""] * args.batch_size)
        print(f"[train] CFG training ON (guidance={args.train_guidance_scale})")
    else:
        train_uncond_emb = None
        train_cond_emb = None
        if args.train_with_cfg:
            print(f"[train] CFG requested but target not available; ignoring")

    def _slice_emb(emb, B):
        """Slice embedding to batch size B. Works for Tensor (SD2) and tuple (SDXL)."""
        if emb is None:
            return None
        if isinstance(emb, tuple):
            return tuple(e[:B] for e in emb)
        return emb[:B]

    for epoch in range(args.epochs):
        if use_dummy:
            iterator = (torch.randn(args.batch_size, 3, args.image_size, args.image_size)
                        for _ in range(args.steps_per_epoch))
        else:
            iterator = iter(loader)

        for batch in iterator:
            img = batch.to(device)
            B = img.shape[0]

            with torch.no_grad():
                # 1) VAE encode
                z0 = target.encode_image(img) if target.available else img
                Hl, Wl = z0.shape[-2:]

                # 2) random mask in pixel space, then downsample to latent
                mask_pix = random_inpaint_mask(B, args.image_size, args.image_size, device)
                mask_z = target.downsample_mask(mask_pix) if target.available else \
                    F.interpolate(mask_pix, size=(Hl, Wl), mode="nearest")

                # 3) cond_latent = masked image encoded
                masked_pix = img * (1 - mask_pix)
                cond_z = target.encode_image(masked_pix) if target.available else masked_pix

                # 4) sample noise and timestep
                eps_gt = torch.randn_like(z0)
                t = sch.sample_timesteps(B)
                z_t = sch.q_sample(z0, eps_gt, t)

                # 5) target eps (frozen forward, no grad)
                # Adjust embeddings if batch size differs (tuple-safe for SDXL)
                cur_uncond = _slice_emb(train_uncond_emb, B)
                cur_cond = _slice_emb(train_cond_emb, B)
                eps_target = target.predict_eps(
                    z_t, t, cond_z, mask_z,
                    cond_emb=cur_cond,
                    uncond_emb=cur_uncond,
                    guidance_scale=args.train_guidance_scale,
                )

            # 6) draft eps (gradients here)
            eps_draft = draft(z_t, t, cond_z, mask_z)

            # 7) loss
            t_norm = sch.t_to_normalized(t)
            loss, logs, M_t, _ = criterion(
                eps_draft, eps_target, eps_gt, z0, mask_z, t_norm
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optim.step()

            # EMA update
            if ema_draft is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_draft.parameters(), draft.parameters()):
                        p_ema.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            if step % args.log_interval == 0:
                print(f"step{step} | loss={loss.item():.4f} "
                      f"l_dist={logs['l_distill']:.4f} l_main={logs['l_main']:.4f} "
                      f"l_unif={logs['l_uniform']:.4f} M_t={logs['M_t_active']:.3f}")

            # Step-interval checkpoint
            if step > 0 and step % args.save_interval == 0:
                ck = {
                    "draft": draft.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                    "args": vars(args),
                }
                if ema_draft is not None:
                    ck["ema_draft"] = ema_draft.state_dict()
                torch.save(ck, os.path.join(args.out_dir, f"draft_step{step:07d}.pt"))
                # Also save as 'latest' for easy access
                torch.save(ck, os.path.join(args.out_dir, "draft_latest.pt"))
                print(f"[train] saved draft_step{step:07d}.pt")

            step += 1
            if step >= args.max_steps:
                break

        # End-of-epoch checkpoint (also)
        ck = {
            "draft": draft.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
        }
        if ema_draft is not None:
            ck["ema_draft"] = ema_draft.state_dict()
        torch.save(ck, os.path.join(args.out_dir, f"draft_e{epoch}.pt"))
        torch.save(ck, os.path.join(args.out_dir, "draft_latest.pt"))
        print(f"[train] saved draft_e{epoch}.pt")
        if step >= args.max_steps:
            break

    # Final save
    ck = {
        "draft": draft.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "args": vars(args),
    }
    if ema_draft is not None:
        ck["ema_draft"] = ema_draft.state_dict()
    torch.save(ck, os.path.join(args.out_dir, "draft_final.pt"))
    print(f"[train] saved draft_final.pt at step {step}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--target_id", type=str,
                   default="stabilityai/stable-diffusion-2-inpainting")
    p.add_argument("--target_dtype", type=str, default="fp32",
                   choices=["fp16", "bf16", "fp32"],
                   help="Target 모델 dtype. SDXL은 'fp16' 추천 (메모리 절약).")
    p.add_argument("--image_size", type=int, default=512,
                   help="SD-Inpainting은 512 권장. SDXL은 1024 권장.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=999,
                   help="크게 설정. max_steps로 실제 제어")
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=5000,
                   help="step 단위 checkpoint 저장 주기")
    p.add_argument("--resume", type=str, default="",
                   help="이어서 학습할 checkpoint 경로")
    # Draft architecture (default = medium, ~50M)
    p.add_argument("--draft_base_ch", type=int, default=128)
    p.add_argument("--draft_ch_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    p.add_argument("--draft_t_dim", type=int, default=512)
    # Loss hyperparams
    p.add_argument("--boundary_weight", type=float, default=1.0)
    p.add_argument("--boundary_kernel", type=int, default=5)
    p.add_argument("--ell", type=float, default=0.3)
    p.add_argument("--alpha_distill", type=float, default=0.5)
    p.add_argument("--gamma_main", type=float, default=2.0)
    p.add_argument("--lambda_uniform", type=float, default=1.0)
    # CFG training
    p.add_argument("--train_with_cfg", action="store_true",
                   help="학습 시 target에 CFG 적용. 학습 비용 2배.")
    p.add_argument("--train_guidance_scale", type=float, default=1.0,
                   help="학습 시 guidance scale. train_with_cfg 없으면 1.0 권장.")
    # EMA
    p.add_argument("--use_ema", action="store_true",
                   help="EMA draft 유지 (안정적 결과)")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())