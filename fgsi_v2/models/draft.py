"""
Draft UNet: 작은 모델, target과 동일한 epsilon-prediction 인터페이스.

핵심: 입출력 channel을 target과 통일
    Input:  [z_t (4) ; mask (1) ; cond_latent (4)] = 9 channels
    Output: eps_hat (4 channels)
Timestep은 integer diffusion step (0 ~ num_train_timesteps-1)을 받음.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DraftEpsUNet(nn.Module):
    """
    Small UNet with epsilon-prediction interface compatible with SD-Inpainting.
    
    Input channels: 9 = 4 (z_t) + 1 (mask) + 4 (cond_latent)
    Output: 4 channels (epsilon prediction)
    
    Default size: ~50M params (medium draft). Adjustable via constructor.
    """
    def __init__(self, latent_ch=4, base_ch=128, ch_mult=(1, 2, 4, 4), t_dim=512,
                 num_train_timesteps=1000):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.latent_ch = latent_ch

        in_ch = latent_ch + 1 + latent_ch  # z_t + mask + cond
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs = [base_ch * m for m in ch_mult]

        self.downs = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.downs.append(nn.ModuleList([
                ResBlock(prev, c, t_dim),
                ResBlock(c, c, t_dim),
                nn.Conv2d(c, c, 3, stride=2, padding=1),
            ]))
            prev = c

        self.mid1 = ResBlock(prev, prev, t_dim)
        self.mid2 = ResBlock(prev, prev, t_dim)

        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(prev, prev, 4, stride=2, padding=1),
                ResBlock(prev + c, c, t_dim),
                ResBlock(c, c, t_dim),
            ]))
            prev = c

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, latent_ch, 3, padding=1)

    def forward(self, z_t, t, cond, mask):
        """
        z_t:   [B, 4, H, W]
        t:     [B] integer in [0, num_train_timesteps)
        cond:  [B, 4, H, W]
        mask:  [B, 1, H, W]
        return: eps_hat [B, 4, H, W]
        """
        x = torch.cat([z_t, mask, cond], dim=1)  # 9 channels
        t_emb = self.t_mlp(timestep_embedding(t.float(), self.t_dim))

        h = self.stem(x)
        skips = []
        for r1, r2, down in self.downs:
            h = r1(h, t_emb)
            h = r2(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up, r1, r2 in self.ups:
            h = up(h)
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], dim=1)
            h = r1(h, t_emb)
            h = r2(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


if __name__ == "__main__":
    net = DraftEpsUNet(latent_ch=4)
    z = torch.randn(2, 4, 32, 32)
    cond = torch.randn(2, 4, 32, 32)
    mask = (torch.rand(2, 1, 32, 32) > 0.7).float()
    t = torch.randint(0, 1000, (2,))
    out = net(z, t, cond, mask)
    print(f"draft eps: {out.shape}")
    print(f"draft params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")