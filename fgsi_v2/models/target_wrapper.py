"""
TargetWrapper: Stable Diffusion (SD1.x/SD2/SDXL) Inpainting을 frozen target으로 wrapping.

설계 원칙:
- Target은 frozen pretrained model
- 통일된 외부 인터페이스: predict_eps(z_t, t, cond, mask, ...) -> eps_hat
- 모델 종류 (SD2 vs SDXL)는 model_index.json 보고 자동 감지
- 외부 코드는 같은 인터페이스로 호출, 내부에서 분기 처리

UNet 입력 채널: 9 = 4 (z_t) + 1 (mask) + 4 (masked_image_latent)
  - SD2 / SDXL 모두 9 채널로 동일
출력: 4 channel noise prediction (ε)

SDXL 특이사항:
  - Dual text encoder (CLIP-L + OpenCLIP-G)
  - Pooled text embedding (add_text_embeds)
  - Time/size conditioning (add_time_ids)
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def _detect_pipeline_class(model_id: str) -> str:
    """Read model_index.json to find pipeline class name. Empty string if unknown."""
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


class TargetWrapper(nn.Module):
    """
    Frozen pretrained inpainting target. Supports SD1.x, SD2, and SDXL inpainting.

    External interface (unified):
        predict_eps(z_t, t, cond_latent, mask_latent,
                    cond_emb=None, uncond_emb=None, guidance_scale=1.0)
        -> eps_hat
    """

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-inpainting",
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self._available = False
        self.is_sdxl = False

        cls_name = _detect_pipeline_class(model_id)
        if cls_name and "XL" in cls_name:
            self.is_sdxl = True

        try:
            if self.is_sdxl:
                from diffusers import StableDiffusionXLInpaintPipeline
                self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    model_id, torch_dtype=dtype,
                )
            else:
                from diffusers import StableDiffusionInpaintPipeline
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, safety_checker=None,
                )
            self.pipe.to(device)

            # 공통 components
            self.unet = self.pipe.unet.eval()
            self.vae = self.pipe.vae.eval()
            self.scheduler_ref = self.pipe.scheduler
            self.tokenizer = self.pipe.tokenizer
            self.text_encoder = self.pipe.text_encoder.eval()

            # SDXL-specific
            if self.is_sdxl:
                self.tokenizer_2 = self.pipe.tokenizer_2
                self.text_encoder_2 = self.pipe.text_encoder_2.eval()
                for p in self.text_encoder_2.parameters():
                    p.requires_grad_(False)

            for m in (self.unet, self.vae, self.text_encoder):
                for p in m.parameters():
                    p.requires_grad_(False)

            self._available = True
            self.latent_ch = 4
            self.vae_scaling = self.vae.config.scaling_factor
            self.vae_ds = 8

            self._uncond_emb = None
            self._uncond_pooled = None  # SDXL only

            try:
                self.default_image_size = int(self.unet.config.sample_size) * self.vae_ds
            except Exception:
                self.default_image_size = 1024 if self.is_sdxl else 512

            model_tag = "SDXL-Inpaint" if self.is_sdxl else "SD-Inpaint"
            print(f"[TargetWrapper:{model_tag}] loaded {model_id}")
        except Exception as e:
            print(f"[TargetWrapper] could not load {model_id}: {e}")
            print(f"[TargetWrapper] falling back to dummy mode (random output)")
            self._available = False
            self.latent_ch = 4
            self.vae_scaling = 0.18215
            self.vae_ds = 8
            self.default_image_size = 512

    @property
    def available(self):
        return self._available

    # ------------------------------------------------------------------ #
    # Text encoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _encode_prompt(self, prompts):
        """SD2: Tensor [B,L,D]. SDXL: tuple (hidden, pooled)."""
        if not self._available:
            return None
        if self.is_sdxl:
            ids1 = self.tokenizer(
                prompts, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            ids2 = self.tokenizer_2(
                prompts, padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            out1 = self.text_encoder(ids1, output_hidden_states=True)
            h1 = out1.hidden_states[-2]
            out2 = self.text_encoder_2(ids2, output_hidden_states=True)
            h2 = out2.hidden_states[-2]
            pooled = out2[0]
            hidden = torch.cat([h1, h2], dim=-1)
            return hidden, pooled
        else:
            tokens = self.tokenizer(
                prompts, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            return self.text_encoder(tokens)[0]

    @torch.no_grad()
    def _get_uncond_embedding(self, batch_size: int):
        """Cached empty-prompt embedding.
        SD2 returns hidden. SDXL returns (hidden, pooled).
        """
        if self.is_sdxl:
            if (self._uncond_emb is None or
                    self._uncond_emb.shape[0] != batch_size):
                hidden, pooled = self._encode_prompt([""] * batch_size)
                self._uncond_emb = hidden
                self._uncond_pooled = pooled
            return self._uncond_emb, self._uncond_pooled
        else:
            if (self._uncond_emb is None or
                    self._uncond_emb.shape[0] != batch_size):
                self._uncond_emb = self._encode_prompt([""] * batch_size)
            return self._uncond_emb

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str, batch_size: int,
                             guidance_scale: float):
        """External helper. Returns embeddings ready for predict_eps.

        SD2: (cond_emb, uncond_emb, use_cfg)
            cond_emb, uncond_emb: Tensor or None
        SDXL: ((cond_hidden, cond_pooled), (uncond_hidden, uncond_pooled), use_cfg)
            cond tuple is None if no CFG
        """
        if not self._available:
            return None, None, False
        use_cfg = guidance_scale > 1.0 and bool(prompt)
        if self.is_sdxl:
            u_hidden, u_pooled = self._get_uncond_embedding(batch_size)
            if use_cfg:
                c_hidden, c_pooled = self._encode_prompt([prompt] * batch_size)
                return (c_hidden, c_pooled), (u_hidden, u_pooled), True
            return None, (u_hidden, u_pooled), False
        else:
            uncond_emb = self._get_uncond_embedding(batch_size)
            if use_cfg:
                cond_emb = self._encode_prompt([prompt] * batch_size)
                return cond_emb, uncond_emb, True
            return None, uncond_emb, False

    # ------------------------------------------------------------------ #
    # SDXL micro-conditioning helper
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_add_time_ids(self, batch_size: int, image_size: int):
        """SDXL micro-conditioning:
        [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        """
        time_ids = torch.tensor(
            [image_size, image_size, 0, 0, image_size, image_size],
            dtype=self.dtype, device=self.device,
        )
        return time_ids.unsqueeze(0).expand(batch_size, -1)

    # ------------------------------------------------------------------ #
    # VAE / mask helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self._available:
            B, _, H, W = image.shape
            z3 = F.interpolate(image, size=(H // 8, W // 8),
                               mode="bilinear", align_corners=False)
            z4 = torch.cat([z3, z3.mean(dim=1, keepdim=True)], dim=1)
            return z4
        z = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
        return z * self.vae_scaling

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self._available:
            B, _, h, w = z.shape
            z3 = z[:, :3]
            return F.interpolate(z3, size=(h * 8, w * 8),
                                 mode="bilinear", align_corners=False).clamp(-1, 1)
        x = self.vae.decode(z.to(self.dtype) / self.vae_scaling).sample
        return x.clamp(-1, 1)

    def downsample_mask(self, mask_pix: torch.Tensor) -> torch.Tensor:
        return F.interpolate(mask_pix, scale_factor=1.0 / 8, mode="nearest")

    # ------------------------------------------------------------------ #
    # Main: predict_eps (unified)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_eps(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond_latent: torch.Tensor,
        mask_latent: torch.Tensor,
        cond_emb=None,           # SD2: tensor or None; SDXL: (hidden, pooled) or None
        uncond_emb=None,         # 동일 형태
        guidance_scale: float = 1.0,
        image_size: int = None,  # SDXL only; defaults to self.default_image_size
    ) -> torch.Tensor:
        if not self._available:
            return torch.randn_like(z_t)

        B = z_t.shape[0]
        if mask_latent.shape[-2:] != z_t.shape[-2:]:
            mask_latent = F.interpolate(mask_latent, size=z_t.shape[-2:], mode="nearest")
        if cond_latent.shape[-2:] != z_t.shape[-2:]:
            cond_latent = F.interpolate(cond_latent, size=z_t.shape[-2:],
                                        mode="bilinear", align_corners=False)

        unet_in = torch.cat([z_t, mask_latent, cond_latent], dim=1).to(self.dtype)

        if self.is_sdxl:
            return self._predict_eps_sdxl(
                unet_in, t, cond_emb, uncond_emb, guidance_scale, B, image_size,
                z_t.dtype,
            )
        else:
            return self._predict_eps_sd(
                unet_in, t, cond_emb, uncond_emb, guidance_scale, B, z_t.dtype,
            )

    def _predict_eps_sd(self, unet_in, t, cond_emb, uncond_emb,
                        guidance_scale, B, out_dtype):
        if guidance_scale > 1.0 and cond_emb is not None:
            unet_in2 = torch.cat([unet_in, unet_in], dim=0)
            t2 = torch.cat([t, t], dim=0) if t.dim() > 0 else t
            emb2 = torch.cat([uncond_emb, cond_emb], dim=0)
            eps_both = self.unet(unet_in2, t2, encoder_hidden_states=emb2).sample
            eps_u, eps_c = eps_both.chunk(2, dim=0)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        else:
            if uncond_emb is None:
                uncond_emb = self._get_uncond_embedding(B)
            eps = self.unet(unet_in, t, encoder_hidden_states=uncond_emb).sample
        return eps.to(out_dtype)

    def _predict_eps_sdxl(self, unet_in, t, cond_emb, uncond_emb,
                          guidance_scale, B, image_size, out_dtype):
        if image_size is None:
            image_size = self.default_image_size
        add_time_ids = self._build_add_time_ids(B, image_size)

        if uncond_emb is None:
            u_hidden, u_pooled = self._get_uncond_embedding(B)
        else:
            u_hidden, u_pooled = uncond_emb

        if guidance_scale > 1.0 and cond_emb is not None:
            c_hidden, c_pooled = cond_emb
            unet_in2 = torch.cat([unet_in, unet_in], dim=0)
            t2 = torch.cat([t, t], dim=0) if t.dim() > 0 else t
            emb2 = torch.cat([u_hidden, c_hidden], dim=0)
            pooled2 = torch.cat([u_pooled, c_pooled], dim=0)
            time_ids2 = torch.cat([add_time_ids, add_time_ids], dim=0)
            added = {"text_embeds": pooled2, "time_ids": time_ids2}
            eps_both = self.unet(
                unet_in2, t2,
                encoder_hidden_states=emb2,
                added_cond_kwargs=added,
            ).sample
            eps_u, eps_c = eps_both.chunk(2, dim=0)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        else:
            added = {"text_embeds": u_pooled, "time_ids": add_time_ids}
            eps = self.unet(
                unet_in, t,
                encoder_hidden_states=u_hidden,
                added_cond_kwargs=added,
            ).sample
        return eps.to(out_dtype)


if __name__ == "__main__":
    tw = TargetWrapper(model_id="dummy/none")
    z = torch.randn(1, 4, 32, 32)
    cond = torch.randn(1, 4, 32, 32)
    mask = (torch.rand(1, 1, 32, 32) > 0.7).float()
    t = torch.tensor([500])
    out = tw.predict_eps(z, t, cond, mask)
    print(f"dummy eps: {out.shape}")