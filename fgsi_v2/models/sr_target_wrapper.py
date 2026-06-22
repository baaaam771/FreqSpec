"""
SRTargetWrapper: Stable Diffusion x4 Upscaler를 frozen target으로 wrapping.

FreqSpec를 inpainting 너머 conditional diffusion 일반으로 확장하기 위한
super-resolution 인스턴스화. SDXL-Inpainting target과 정확히 평행한 구조:
frozen pretrained latent-diffusion conditional model을 그대로 verifier로 둔다.

핵심 차이 (inpainting 대비):
- 조건 c = low-resolution image (mask가 아니라 LR RGB)
- 생성 영역 = 전체 latent field (mask 없음 -> region = all-ones)
- UNet 입력 채널: 7 = 4 (z_t) + 3 (low-res RGB, latent resolution)
- noise_level 조건: LR을 low-res scheduler로 noise_level만큼 noising 후
  noise_level 정수를 class_labels로 UNet에 전달 (x4 upscaler 고유 메커니즘)
- VAE: upscaler 전용 (scaling_factor=0.08333), decode 시 x4 upsampling

통일 인터페이스 (speculative_general.fgsr_refine 계약):
    predict_eps(z_t, t, cond_lr, region, cond_emb, uncond_emb,
                guidance_scale, noise_level=20) -> eps_hat [B,4,h,w]
region 인자는 인터페이스 평행성을 위해 받지만 사용하지 않는다 (전체 필드 생성).

※ 서버에서 실제 가중치로 검증할 항목 (이 환경에서는 dummy로만 스모크):
  - unet.config.in_channels == 7
  - scheduler prediction_type == "epsilon" (DDIM step이 epsilon 가정)
  - vae.config.scaling_factor (보통 0.08333), VAE decode가 x4 upsample
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def _detect_pipeline_class(model_id: str) -> str:
    if not os.path.isdir(model_id):
        return ""
    idx = os.path.join(model_id, "model_index.json")
    if os.path.isfile(idx):
        try:
            with open(idx, "r") as f:
                info = json.load(f)
            return info.get("_class_name", "")
        except Exception:
            return ""
    return ""


class SRTargetWrapper(nn.Module):
    """Frozen pretrained x4 latent-diffusion super-resolution target."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
                 dtype=torch.float32, device="cpu", default_noise_level: int = 20):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.default_noise_level = default_noise_level
        self._available = False
        self._uncond_emb = None

        try:
            from diffusers import StableDiffusionUpscalePipeline
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, torch_dtype=dtype,
            )
            self.pipe.to(device)
            self.unet = self.pipe.unet.eval()
            self.vae = self.pipe.vae.eval()
            self.scheduler_ref = self.pipe.scheduler
            self.low_res_scheduler = self.pipe.low_res_scheduler
            self.tokenizer = self.pipe.tokenizer
            self.text_encoder = self.pipe.text_encoder.eval()

            for m in (self.unet, self.vae, self.text_encoder):
                for p in m.parameters():
                    p.requires_grad_(False)

            self._available = True
            self.latent_ch = 4
            self.vae_scaling = getattr(self.vae.config, "scaling_factor", 0.08333)
            # x4 upscaler: latent operates at LR resolution; VAE decode upsamples x4.
            self.vae_upscale = 4
            pred = getattr(self.scheduler_ref.config, "prediction_type", "epsilon")
            self.prediction_type = pred
            # The generalized sampler works in epsilon space. For v-prediction
            # targets (e.g. the x4 upscaler) we convert the UNet output to
            # epsilon inside predict_eps using the model's own alphas_cumprod.
            self.alphas_cumprod = self.scheduler_ref.alphas_cumprod.to(device)
            if pred not in ("epsilon", "v_prediction"):
                print(f"[SRTargetWrapper] WARNING: prediction_type={pred} "
                      f"unhandled (only epsilon / v_prediction supported)")
            else:
                print(f"[SRTargetWrapper] prediction_type={pred} "
                      f"({'v->eps conversion ON' if pred == 'v_prediction' else 'native eps'})")
            in_ch = getattr(self.unet.config, "in_channels", None)
            if in_ch is not None and in_ch != 7:
                print(f"[SRTargetWrapper] WARNING: unet.in_channels={in_ch} "
                      f"(expected 7 for x4 upscaler)")
            print(f"[SRTargetWrapper:SDx4-Upscaler] loaded {model_id}")
        except Exception as e:
            print(f"[SRTargetWrapper] could not load {model_id}: {e}")
            print(f"[SRTargetWrapper] falling back to dummy mode (random output)")
            self._available = False
            self.latent_ch = 4
            self.vae_scaling = 0.08333
            self.vae_upscale = 4
            self.prediction_type = "epsilon"
            self.alphas_cumprod = None

    @property
    def available(self):
        return self._available

    # ------------------------------------------------------------------ #
    # Text encoding (single CLIP encoder, SD2-style)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _encode_prompt(self, prompts):
        if not self._available:
            return None
        tokens = self.tokenizer(
            prompts, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0]

    @torch.no_grad()
    def _get_uncond_embedding(self, batch_size: int):
        if (self._uncond_emb is None or self._uncond_emb.shape[0] != batch_size):
            self._uncond_emb = self._encode_prompt([""] * batch_size)
        return self._uncond_emb

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str, batch_size: int, guidance_scale: float):
        if not self._available:
            return None, None, False
        use_cfg = guidance_scale > 1.0 and bool(prompt)
        uncond_emb = self._get_uncond_embedding(batch_size)
        if use_cfg:
            cond_emb = self._encode_prompt([prompt] * batch_size)
            return cond_emb, uncond_emb, True
        return None, uncond_emb, False

    # ------------------------------------------------------------------ #
    # VAE / LR-conditioning helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """HR image (pixel, [-1,1]) -> latent. Used for q_sample targets in training."""
        if not self._available:
            B, _, H, W = image.shape
            z3 = F.interpolate(image, size=(H // self.vae_upscale, W // self.vae_upscale),
                               mode="bilinear", align_corners=False)
            return torch.cat([z3, z3.mean(dim=1, keepdim=True)], dim=1)
        z = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
        return z * self.vae_scaling

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self._available:
            B, _, h, w = z.shape
            z3 = z[:, :3]
            return F.interpolate(z3, size=(h * self.vae_upscale, w * self.vae_upscale),
                                 mode="bilinear", align_corners=False).clamp(-1, 1)
        x = self.vae.decode(z.to(self.dtype) / self.vae_scaling).sample
        return x.clamp(-1, 1)

    @torch.no_grad()
    def prepare_lr_cond(self, lr_image: torch.Tensor, noise_level: int = None):
        """Noise the LR conditioning image once (x4-upscaler convention).

        lr_image: [B,3,h,w] in [-1,1] at *latent* spatial resolution.
        Returns (cond_lr_noised [B,3,h,w], noise_level_tensor [B]).
        The LR anchor enters the model only through this conditioning, so no
        known-region blending is needed at sampling time.
        """
        nl = self.default_noise_level if noise_level is None else noise_level
        B = lr_image.shape[0]
        nl_tensor = torch.full((B,), int(nl), device=lr_image.device, dtype=torch.long)
        if not self._available:
            return lr_image, nl_tensor
        noise = torch.randn_like(lr_image)
        cond = self.low_res_scheduler.add_noise(lr_image.to(self.dtype), noise, nl_tensor)
        return cond, nl_tensor

    # ------------------------------------------------------------------ #
    # Main: predict_eps (unified interface)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_eps(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond_lr: torch.Tensor,        # [B,3,h,w] noised LR (from prepare_lr_cond)
        region: torch.Tensor = None,  # unused (whole field); interface parity only
        cond_emb=None,
        uncond_emb=None,
        guidance_scale: float = 1.0,
        noise_level=None,             # int or [B] long tensor
    ) -> torch.Tensor:
        if not self._available:
            return torch.randn_like(z_t)

        B = z_t.shape[0]
        if cond_lr.shape[-2:] != z_t.shape[-2:]:
            cond_lr = F.interpolate(cond_lr, size=z_t.shape[-2:],
                                    mode="bilinear", align_corners=False)
        if noise_level is None:
            noise_level = torch.full((B,), self.default_noise_level,
                                     device=z_t.device, dtype=torch.long)
        elif not torch.is_tensor(noise_level):
            noise_level = torch.full((B,), int(noise_level),
                                     device=z_t.device, dtype=torch.long)

        unet_in = torch.cat([z_t, cond_lr], dim=1).to(self.dtype)  # 7 channels

        if guidance_scale > 1.0 and cond_emb is not None:
            unet_in2 = torch.cat([unet_in, unet_in], dim=0)
            t2 = torch.cat([t, t], dim=0) if t.dim() > 0 else t
            emb2 = torch.cat([uncond_emb, cond_emb], dim=0)
            nl2 = torch.cat([noise_level, noise_level], dim=0)
            out_both = self.unet(
                unet_in2, t2, encoder_hidden_states=emb2, class_labels=nl2,
            ).sample
            out_u, out_c = out_both.chunk(2, dim=0)
            model_out = out_u + guidance_scale * (out_c - out_u)
        else:
            if uncond_emb is None:
                uncond_emb = self._get_uncond_embedding(B)
            model_out = self.unet(
                unet_in, t, encoder_hidden_states=uncond_emb, class_labels=noise_level,
            ).sample

        # Convert to epsilon space if the target is a v-prediction model.
        # CFG is affine in the model output, so converting the combined output
        # is equivalent to converting each branch:  eps = sqrt(ab)*v + sqrt(1-ab)*z
        if self.prediction_type == "v_prediction":
            ac = self.alphas_cumprod[t].to(model_out.dtype).view(-1, 1, 1, 1)
            sqrt_ac = ac.sqrt()
            sqrt_1m = (1.0 - ac).sqrt()
            eps = sqrt_ac * model_out + sqrt_1m * z_t.to(model_out.dtype)
        else:
            eps = model_out
        return eps.to(z_t.dtype)


if __name__ == "__main__":
    tw = SRTargetWrapper(model_id="dummy/none")
    z = torch.randn(1, 4, 32, 32)
    lr = torch.randn(1, 3, 32, 32)
    t = torch.tensor([500])
    cond, nl = tw.prepare_lr_cond(lr, noise_level=20)
    out = tw.predict_eps(z, t, cond, noise_level=nl)
    print(f"dummy SR eps: {out.shape}  noise_level={nl.tolist()}")