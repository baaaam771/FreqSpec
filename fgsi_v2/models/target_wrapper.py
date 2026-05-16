"""
TargetWrapper: Stable Diffusion Inpainting을 frozen target으로 wrapping.

설계 원칙:
- Target은 frozen pretrained model
- 통일된 인터페이스: predict_eps(z_t, t, cond, mask) -> eps_hat
- 본 wrapper는 model-agnostic하게 설계되어 다른 latent diffusion inpainting 모델 추가 가능
- diffusers의 StableDiffusionInpaintPipeline의 UNet을 사용
- Text conditioning은 빈 prompt(unconditional) 또는 placeholder 사용 (inpainting은 본질적으로 text 없이도 작동 가능)

SD-Inpainting UNet 입력 채널: 9 = 4 (z_t) + 1 (mask) + 4 (masked_image_latent)
출력: 4 channel noise prediction (ε)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetWrapper(nn.Module):
    """
    Frozen pretrained inpainting target.
    
    Forward signature (unified):
        predict_eps(z_t, t, cond_latent, mask_latent) -> eps_hat
    """
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-inpainting",
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self._available = False

        try:
            from diffusers import StableDiffusionInpaintPipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None,
            )
            self.pipe.to(device)
            # UNet, VAE, text encoder를 frozen으로 둠
            self.unet = self.pipe.unet.eval()
            self.vae = self.pipe.vae.eval()
            self.tokenizer = self.pipe.tokenizer
            self.text_encoder = self.pipe.text_encoder.eval()
            self.scheduler_ref = self.pipe.scheduler  # noise schedule 참고용
            for m in (self.unet, self.vae, self.text_encoder):
                for p in m.parameters():
                    p.requires_grad_(False)
            self._available = True
            self.latent_ch = 4
            self.vae_scaling = self.vae.config.scaling_factor
            self.vae_ds = 8
            # SD-Inpainting의 UNet은 9채널 입력
            self._uncond_emb = None  # lazy
            print(f"[TargetWrapper] loaded {model_id}")
        except Exception as e:
            print(f"[TargetWrapper] could not load {model_id}: {e}")
            print(f"[TargetWrapper] falling back to dummy mode (random output)")
            self._available = False
            self.latent_ch = 4
            self.vae_scaling = 0.18215
            self.vae_ds = 8

    @property
    def available(self):
        return self._available

    @torch.no_grad()
    def _encode_prompt(self, prompts):
        """Tokenize and encode a list of prompts."""
        if not self._available:
            return None
        tokens = self.tokenizer(
            prompts,
            padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0]

    @torch.no_grad()
    def _get_uncond_embedding(self, batch_size: int):
        """Cached empty-prompt text embedding (for unconditional branch / no-CFG)."""
        if self._uncond_emb is None or self._uncond_emb.shape[0] != batch_size:
            self._uncond_emb = self._encode_prompt([""] * batch_size)
        return self._uncond_emb

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str, batch_size: int, guidance_scale: float):
        """
        Returns:
            cond_emb:   [B, L, D] conditional text embedding (or None if CFG off)
            uncond_emb: [B, L, D] unconditional (empty prompt) embedding
            use_cfg: bool
        """
        if not self._available:
            return None, None, False
        uncond_emb = self._get_uncond_embedding(batch_size)
        if guidance_scale > 1.0 and prompt:
            cond_emb = self._encode_prompt([prompt] * batch_size)
            return cond_emb, uncond_emb, True
        return None, uncond_emb, False

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image in [-1,1] -> latent. [B,3,H,W] -> [B,4,H/8,W/8]"""
        if not self._available:
            return image  # dummy
        z = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
        return z * self.vae_scaling

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return z
        x = self.vae.decode(z.to(self.dtype) / self.vae_scaling).sample
        return x.clamp(-1, 1)

    def downsample_mask(self, mask_pix: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] mask -> latent resolution by nearest."""
        if self.vae_ds == 1:
            return mask_pix
        return F.interpolate(mask_pix, scale_factor=1.0 / self.vae_ds, mode="nearest")

    @torch.no_grad()
    def predict_eps(
        self,
        z_t: torch.Tensor,           # [B,4,Hl,Wl]
        t: torch.Tensor,              # [B] integer diffusion timesteps
        cond_latent: torch.Tensor,    # [B,4,Hl,Wl] masked-image latent
        mask_latent: torch.Tensor,    # [B,1,Hl,Wl] (1=hole)
        cond_emb: torch.Tensor = None,    # text embedding (for CFG-conditional branch)
        uncond_emb: torch.Tensor = None,  # text embedding (for unconditional branch)
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Stable Diffusion Inpainting 9-channel input:
            [z_t (4) ; mask (1) ; masked_image_latent (4)]

        If guidance_scale > 1.0 and cond_emb is provided, uses classifier-free guidance:
            eps_final = uncond + guidance * (cond - uncond)
        Otherwise unconditional only.
        """
        if not self._available:
            return torch.randn_like(z_t)

        B = z_t.shape[0]
        # mask, cond_latent를 latent 해상도에 맞춤
        if mask_latent.shape[-2:] != z_t.shape[-2:]:
            mask_latent = F.interpolate(mask_latent, size=z_t.shape[-2:], mode="nearest")
        if cond_latent.shape[-2:] != z_t.shape[-2:]:
            cond_latent = F.interpolate(cond_latent, size=z_t.shape[-2:],
                                        mode="bilinear", align_corners=False)

        unet_in = torch.cat([z_t, mask_latent, cond_latent], dim=1).to(self.dtype)

        if guidance_scale > 1.0 and cond_emb is not None:
            # CFG: 두 번 forward
            unet_in2 = torch.cat([unet_in, unet_in], dim=0)
            t2 = torch.cat([t, t], dim=0) if t.dim() > 0 else t
            emb2 = torch.cat([uncond_emb, cond_emb], dim=0)
            eps_both = self.unet(unet_in2, t2, encoder_hidden_states=emb2).sample
            eps_uncond, eps_cond = eps_both.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            if uncond_emb is None:
                uncond_emb = self._get_uncond_embedding(B)
            eps = self.unet(unet_in, t, encoder_hidden_states=uncond_emb).sample

        return eps.to(z_t.dtype)


if __name__ == "__main__":
    # dummy mode test
    tw = TargetWrapper(model_id="dummy/none")
    z = torch.randn(1, 4, 32, 32)
    cond = torch.randn(1, 4, 32, 32)
    mask = (torch.rand(1, 1, 32, 32) > 0.7).float()
    t = torch.tensor([500])
    out = tw.predict_eps(z, t, cond, mask)
    print(f"dummy eps: {out.shape}")