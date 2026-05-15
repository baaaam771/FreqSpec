"""
DDPM/DDIM noise scheduler compatible with SD-Inpainting.

설계:
- Linear or scaled-linear β schedule (SD 표준: scaled_linear)
- ε-prediction
- 학습 시: q_sample (forward noising)
- 추론 시: DDIM sampling (deterministic, 적은 step 수로 가능)

Target과 같은 noise schedule을 써야 ε 예측이 서로 비교 가능.
SD-Inpainting의 기본은 scaled_linear, num_train_timesteps=1000, β_start=0.00085, β_end=0.012.
"""
import torch


class DDPMSchedule:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        device: str = "cpu",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.device = device

        if beta_schedule == "scaled_linear":
            betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
                dtype=torch.float64
            ) ** 2
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps,
                                   dtype=torch.float64)
        else:
            raise ValueError(beta_schedule)

        self.betas = betas.float().to(device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(0, self.num_train_timesteps,
                             (batch_size,), device=self.device, dtype=torch.long)

    def q_sample(self, z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        """
        z_t = √α̅_t · z_0 + √(1-α̅_t) · ε
        t: [B] integer timesteps
        """
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        som = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sa * z0 + som * eps

    def t_to_normalized(self, t: torch.Tensor):
        """integer t in [0, T-1] -> normalized [0, 1] for LWD masking."""
        return t.float() / (self.num_train_timesteps - 1)

    def get_ddim_schedule(self, num_inference_steps: int):
        """
        Return DDIM step sequence: descending integer timesteps.
        e.g. T=1000, n=50 -> [999, 979, 959, ..., 19, -1+0?]
        """
        step = self.num_train_timesteps // num_inference_steps
        ts = torch.arange(0, self.num_train_timesteps, step, device=self.device).flip(0)
        return ts  # [num_inference_steps]

    def ddim_step(
        self,
        z_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ):
        """
        One DDIM denoising step. t > t_prev (going from noisy to clean).
        Deterministic if eta=0.
        """
        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

        # predicted z0
        z0_pred = (z_t - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)

        # direction to z_t_prev
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
        dir_zt = torch.sqrt(1 - a_prev - sigma ** 2) * eps_pred

        z_prev = torch.sqrt(a_prev) * z0_pred + dir_zt
        if eta > 0:
            z_prev = z_prev + sigma * torch.randn_like(z_t)
        return z_prev, z0_pred


if __name__ == "__main__":
    sch = DDPMSchedule()
    z0 = torch.randn(2, 4, 32, 32)
    eps = torch.randn(2, 4, 32, 32)
    t = sch.sample_timesteps(2)
    z_t = sch.q_sample(z0, eps, t)
    print(f"q_sample: {z_t.shape}, t={t.tolist()}")
    print(f"DDIM schedule (50 steps): first 5 = {sch.get_ddim_schedule(50)[:5].tolist()}")
