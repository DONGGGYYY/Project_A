"""
Diffusion Policy implementation for LIBERO.

参考: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action
Diffusion", RSS 2023. https://github.com/columbia-ai-robotics/diffusion_policy

核心思想:
- 不直接预测 action，而是学习从噪声到 action 序列的去噪过程 (DDPM)
- 条件: 图像观测的 ResNet 特征 (FiLM 调制注入)
- 架构: 1D Conditional U-Net 处理 action sequence (B, T, action_dim)
- 推理: DDIM 加速采样 (训练 100 步 -> 推理 10 步)

形状约定:
- images: (B, 3, H, W)        -- 单帧观测 (取 horizon 中第一帧条件)
- actions: (B, T, action_dim) -- 长度 T 的动作序列
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# 1. Vision Encoder
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """ResNet18 -> 512-dim feature. 复用 baseline 经验。"""

    def __init__(self, out_dim: int = 256, pretrained: bool = False):
        super().__init__()
        backbone = resnet18(pretrained=pretrained)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W) -> (B, out_dim)
        feat = self.backbone(images)
        return self.proj(feat)


# ---------------------------------------------------------------------------
# 2. Sinusoidal time embedding (扩散步 t 编码)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int -> (B, dim)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# 3. Conditional 1D U-Net building blocks
# ---------------------------------------------------------------------------

class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    1D residual block，FiLM 注入 (cond_dim -> out_ch * 2) 调制 GroupNorm 之后的特征。
    cond 同时包含时间步和图像特征。
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size)

        # FiLM: 输出 scale + shift
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )

        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        out = self.block1(x)
        embed = self.cond_encoder(cond)  # (B, 2*C)
        scale, shift = embed.chunk(2, dim=-1)
        out = out * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        out = self.block2(out)
        return out + self.residual(x)


# ---------------------------------------------------------------------------
# 4. Conditional U-Net 1D (主网络)
# ---------------------------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    """
    1D U-Net，作用在动作序列上 (B, action_dim, T)。
    cond = [time_emb, image_feat] -> FiLM 注入每个 residual block。
    """

    def __init__(
        self,
        action_dim: int = 7,
        cond_dim: int = 256,           # image feature dim
        time_dim: int = 128,
        down_dims: tuple = (64, 128, 256),
        kernel_size: int = 3,
    ):
        super().__init__()

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        full_cond_dim = time_dim + cond_dim
        all_dims = (action_dim,) + tuple(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = down_dims[-1]

        # 下采样
        self.down_modules = nn.ModuleList()
        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(d_in, d_out, full_cond_dim, kernel_size),
                    ConditionalResidualBlock1D(d_out, d_out, full_cond_dim, kernel_size),
                    nn.Conv1d(d_out, d_out, 3, stride=2, padding=1) if not is_last else nn.Identity(),
                ])
            )

        # 中间
        self.mid_block1 = ConditionalResidualBlock1D(mid_dim, mid_dim, full_cond_dim, kernel_size)
        self.mid_block2 = ConditionalResidualBlock1D(mid_dim, mid_dim, full_cond_dim, kernel_size)

        # 上采样 (镜像): 每层都做真正的 ConvTranspose 以恢复序列长度
        # 下采样了 len(in_out)-1 次，所以上采样也要 len(in_out)-1 次
        self.up_modules = nn.ModuleList()
        for ind, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(d_out * 2, d_in, full_cond_dim, kernel_size),
                    ConditionalResidualBlock1D(d_in, d_in, full_cond_dim, kernel_size),
                    nn.ConvTranspose1d(d_in, d_in, 4, stride=2, padding=1),
                ])
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,      # (B, T, action_dim) noisy action sequence
        timestep: torch.Tensor,    # (B,) int
        cond: torch.Tensor,        # (B, cond_dim) image feature
    ) -> torch.Tensor:
        # -> (B, action_dim, T)
        x = sample.transpose(1, 2)

        t_emb = self.time_mlp(timestep)
        full_cond = torch.cat([t_emb, cond], dim=-1)

        # Down
        skips = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, full_cond)
            x = resnet2(x, full_cond)
            skips.append(x)
            x = downsample(x)

        # Mid
        x = self.mid_block1(x, full_cond)
        x = self.mid_block2(x, full_cond)

        # Up
        for resnet1, resnet2, upsample in self.up_modules:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet1(x, full_cond)
            x = resnet2(x, full_cond)
            x = upsample(x)

        # final
        x = self.final_conv(x)
        return x.transpose(1, 2)  # (B, T, action_dim)


# ---------------------------------------------------------------------------
# 5. DDPM scheduler (训练前向加噪 + DDIM 推理采样)
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """
    极简版本 DDPM/DDIM scheduler。
    - num_train_timesteps 步训练
    - DDIM 采样在 num_inference_timesteps 步内完成
    - beta schedule: 线性
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(
        self,
        original: torch.Tensor,         # (B, T, A) clean action
        noise: torch.Tensor,            # 同 shape
        timesteps: torch.Tensor,        # (B,) int
    ) -> torch.Tensor:
        """前向加噪: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps"""
        alpha_bar = self.alphas_cumprod[timesteps]  # (B,)
        sqrt_a = alpha_bar.sqrt().view(-1, 1, 1)
        sqrt_1ma = (1 - alpha_bar).sqrt().view(-1, 1, 1)
        return sqrt_a * original + sqrt_1ma * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,                       # callable(noisy, t, cond) -> pred_noise
        cond: torch.Tensor,             # (B, cond_dim)
        shape: tuple,                   # (B, T, action_dim)
        num_inference_steps: int = 10,
        eta: float = 0.0,               # 0 -> deterministic DDIM
    ) -> torch.Tensor:
        """DDIM 采样，从纯噪声去噪到动作序列。"""
        device = self.device
        x = torch.randn(shape, device=device)

        # 选取均匀间隔的 timesteps，从大到小
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0])
            pred_noise = model_fn(x, t_batch, cond)

            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # x_0 预测
            pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            # 方向项
            dir_xt = (1 - alpha_bar_prev).sqrt() * pred_noise
            x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt

        return x


# ---------------------------------------------------------------------------
# 6. Diffusion Policy 包装类
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    完整 Diffusion Policy。

    用法:
        policy = DiffusionPolicy(action_dim=7, horizon=16)
        # 训练
        loss = policy.compute_loss(images, actions)
        loss.backward()
        # 推理
        actions = policy.sample(images)  # (B, horizon, action_dim)
    """

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 16,
        cond_dim: int = 256,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_inference_steps = num_inference_steps

        self.vision_encoder = VisionEncoder(out_dim=cond_dim)
        self.unet = ConditionalUnet1D(
            action_dim=action_dim,
            cond_dim=cond_dim,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def _encode_obs(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) 或 (B, T, 3, H, W) -> 取第一帧"""
        if images.ndim == 5:
            images = images[:, 0]
        return self.vision_encoder(images)

    def compute_loss(
        self,
        images: torch.Tensor,    # (B, 3, H, W) 或 (B, T, 3, H, W)
        actions: torch.Tensor,   # (B, T, action_dim)
    ) -> torch.Tensor:
        """DDPM 训练 loss: MSE(预测噪声, 真实噪声)"""
        device = actions.device
        B = actions.shape[0]

        cond = self._encode_obs(images)

        # 随机时间步
        t = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,), device=device
        ).long()
        noise = torch.randn_like(actions)
        noisy = self.scheduler.add_noise(actions, noise, t)

        pred_noise = self.unet(noisy, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,                # (B, 3, H, W) 或 (B, T, 3, H, W)
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """DDIM 推理: 输入观测，输出 (B, horizon, action_dim)"""
        steps = num_inference_steps or self.num_inference_steps
        cond = self._encode_obs(images)
        B = cond.shape[0]
        shape = (B, self.horizon, self.action_dim)
        return self.scheduler.ddim_sample(self.unet, cond, shape, num_inference_steps=steps)


# ---------------------------------------------------------------------------
# 7. Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, T, A = 4, 16, 7
    H, W = 128, 128

    images = torch.randn(B, 3, H, W, device=device)
    actions = torch.randn(B, T, A, device=device)

    policy = DiffusionPolicy(action_dim=A, horizon=T).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")

    loss = policy.compute_loss(images, actions)
    print(f"Train loss (random data): {loss.item():.4f}")

    sampled = policy.sample(images)
    print(f"Sampled actions shape: {sampled.shape}")
    print("OK")
