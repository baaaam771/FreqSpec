"""
Draft 모델 학습 loss.

설계 의도:
- 평탄 영역 (1 - M_t): target eps를 모방 (distillation) → 추론 시 accept rate ↑
- 어려운 영역 (M_t): GT eps를 직접 학습 → 가끔 reject되어도 합리적
- M_t는 LWD-style time-dependent binary mask (boundary-aware)

L_draft = (1 - M_t) ⊙ α_distill ||ε_draft - ε_target||²        # 평탄: target 모방
        + M_t ⊙ γ_main      ||ε_draft - ε_gt    ||²            # 어려운: GT 학습
        + λ_uniform         ||ε_draft - ε_gt    ||²            # 안전망: 전 영역 약한 학습

α_distill, γ_main, λ_uniform은 hyperparameter.
실용적으로는 평탄 영역에서 distillation을 가장 강하게.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.wavelet import DWT2D, combined_saliency, lwd_time_mask


class DraftLoss:
    def __init__(
        self,
        wavelet: str = "haar",
        boundary_weight: float = 1.0,
        boundary_kernel: int = 5,
        ell: float = 0.3,
        alpha_distill: float = 1.0,   # 평탄 영역에서 target 모방 강도
        gamma_main: float = 1.0,      # 어려운 영역에서 GT 학습 강도
        lambda_uniform: float = 0.1,  # 전 영역 안전망
        device: str = "cpu",
    ):
        self.dwt = DWT2D(wavelet).to(device)
        self.boundary_weight = boundary_weight
        self.boundary_kernel = boundary_kernel
        self.ell = ell
        self.alpha_distill = alpha_distill
        self.gamma_main = gamma_main
        self.lambda_uniform = lambda_uniform

    def to(self, device):
        self.dwt = self.dwt.to(device)
        return self

    @torch.no_grad()
    def compute_mask(self, z0, mask_z, t_normalized):
        """Compute A_combined and binary M_t."""
        sal = combined_saliency(
            z0, mask_z, self.dwt,
            boundary_weight=self.boundary_weight,
            boundary_kernel=self.boundary_kernel,
            target_size=z0.shape[-2:],
        )
        M_t = lwd_time_mask(sal, t_normalized, ell=self.ell)
        return M_t, sal

    def __call__(
        self,
        eps_draft: torch.Tensor,      # [B,4,Hl,Wl]
        eps_target: torch.Tensor,     # [B,4,Hl,Wl] (no grad)
        eps_gt: torch.Tensor,         # [B,4,Hl,Wl]
        z0: torch.Tensor,             # [B,4,Hl,Wl]
        mask_z: torch.Tensor,         # [B,1,Hl,Wl]
        t_normalized: torch.Tensor,   # [B] in [0,1]
    ):
        M_t, sal = self.compute_mask(z0, mask_z, t_normalized)  # [B,1,H,W]

        # 각 위치별 squared error (channel mean)
        err_distill = (eps_draft - eps_target.detach()).pow(2).mean(dim=1, keepdim=True)
        err_main = (eps_draft - eps_gt).pow(2).mean(dim=1, keepdim=True)

        l_distill = (self.alpha_distill * (1 - M_t) * err_distill).mean()
        l_main = (self.gamma_main * M_t * err_main).mean()
        l_uniform = self.lambda_uniform * err_main.mean()

        total = l_distill + l_main + l_uniform
        return total, {
            "l_distill": l_distill.item(),
            "l_main": l_main.item(),
            "l_uniform": l_uniform.item(),
            "M_t_active": M_t.mean().item(),
        }, M_t, sal


if __name__ == "__main__":
    crit = DraftLoss()
    z0 = torch.randn(2, 4, 32, 32)
    eps_gt = torch.randn(2, 4, 32, 32)
    eps_target = eps_gt + 0.1 * torch.randn_like(eps_gt)
    eps_draft = eps_gt + 0.3 * torch.randn_like(eps_gt)
    mask = (torch.rand(2, 1, 32, 32) > 0.7).float()
    t_norm = torch.rand(2)
    loss, logs, M_t, sal = crit(eps_draft, eps_target, eps_gt, z0, mask, t_norm)
    print(f"loss: {loss.item():.4f}")
    print(f"logs: {logs}")
