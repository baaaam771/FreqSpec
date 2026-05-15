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
    def _get_uncond_embedding(self, batch_size: int):
        """Cached empty-prompt text embedding for unconditional inpainting."""
        if self._uncond_emb is None or self._uncond_emb.shape[0] != batch_size:
            tokens = self.tokenizer(
                [""] * batch_size,
                padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            self._uncond_emb = self.text_encoder(tokens)[0]
        return self._uncond_emb

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image in [-1,1] -> latent. [B,3,H,W] -> [B,4,H/8,W/8]"""
        if not self._available:
            # Dummy mode: 가짜 latent (3채널 -> 4채널, /8 다운샘플)
            B, _, H, W = image.shape
            z = F.interpolate(image, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
            # 채널을 3 -> 4로 확장 (간단히 평균을 4번째 채널로)
            z4 = torch.cat([z, z.mean(dim=1, keepdim=True)], dim=1)
            return z4
        z = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
        return z * self.vae_scaling

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self._available:
            # Dummy: 4채널 latent -> 3채널 픽셀, 8배 업샘플
            B, _, h, w = z.shape
            z3 = z[:, :3]  # 첫 3채널만
            return F.interpolate(z3, size=(h * 8, w * 8), mode="bilinear", align_corners=False).clamp(-1, 1)
        x = self.vae.decode(z.to(self.dtype) / self.vae_scaling).sample
        return x.clamp(-1, 1)

    def downsample_mask(self, mask_pix: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] mask -> latent resolution by nearest."""
        # 항상 8배 다운샘플 (dummy든 진짜든 통일)
        return F.interpolate(mask_pix, scale_factor=1.0 / 8, mode="nearest")
    
    @torch.no_grad()
    def predict_eps(
        self,
        z_t: torch.Tensor,           # [B,4,Hl,Wl]
        t: torch.Tensor,              # [B] integer diffusion timesteps
        cond_latent: torch.Tensor,    # [B,4,Hl,Wl] masked-image latent
        mask_latent: torch.Tensor,    # [B,1,Hl,Wl] (1=hole)
    ) -> torch.Tensor:
        """
        Stable Diffusion Inpainting 9-channel input:
            [z_t (4) ; mask (1) ; masked_image_latent (4)]
        """
        if not self._available:
            # dummy: 임의의 일관된 epsilon 반환 (개발 환경용)
            return torch.randn_like(z_t)

        B = z_t.shape[0]
        # mask와 cond_latent를 latent 해상도에 맞춤
        if mask_latent.shape[-2:] != z_t.shape[-2:]:
            mask_latent = F.interpolate(mask_latent, size=z_t.shape[-2:], mode="nearest")
        if cond_latent.shape[-2:] != z_t.shape[-2:]:
            cond_latent = F.interpolate(cond_latent, size=z_t.shape[-2:], mode="bilinear", align_corners=False)

        unet_in = torch.cat([z_t, mask_latent, cond_latent], dim=1).to(self.dtype)
        emb = self._get_uncond_embedding(B)
        eps = self.unet(unet_in, t, encoder_hidden_states=emb).sample
        return eps.to(z_t.dtype)


if __name__ == "__main__":
    import sys
    device = "cuda" if len(sys.argv) > 1 and sys.argv[1] == "real" else "cpu"
    if device == "cuda":
        # 실제 SD-Inpainting 로드 (5GB 다운로드)
        tw = TargetWrapper(
            model_id="stabilityai/stable-diffusion-2-inpainting",
            device=device,
        )
    else:
        # dummy 테스트
        tw = TargetWrapper(model_id="dummy/none")
    
    z = torch.randn(1, 4, 32, 32, device=device)
    cond = torch.randn(1, 4, 32, 32, device=device)
    mask = (torch.rand(1, 1, 32, 32, device=device) > 0.7).float()
    t = torch.tensor([500], device=device)
    out = tw.predict_eps(z, t, cond, mask)
    print(f"eps: {out.shape}, available={tw.available}")

# if __name__ == "__main__":
#     # dummy mode test
#     tw = TargetWrapper(model_id="dummy/none")
#     z = torch.randn(1, 4, 32, 32)
#     cond = torch.randn(1, 4, 32, 32)
#     mask = (torch.rand(1, 1, 32, 32) > 0.7).float()
#     t = torch.tensor([500])
#     out = tw.predict_eps(z, t, cond, mask)
#     print(f"dummy eps: {out.shape}")