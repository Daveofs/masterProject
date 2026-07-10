"""Time-conditioned UNet for conditional flow matching.

Unlike model.py's UNet (deterministic, low -> high directly), this predicts a
velocity field v_theta(x_t, t) used to integrate an ODE from x_0 = low_log
(the low-fidelity patch, in log1p-delta space - NOT noise) to x_1 = high_log,
along the straight-line interpolant x_t = (1-t)*x_0 + t*x_1.

Starting the flow from the low-fidelity map instead of Gaussian noise is the
whole point here: x_0 is already close to x_1 (same large-scale structure,
only small-scale correction differs), so the ODE trajectory is close to
straight and should need very few integration steps at inference - unlike a
typical diffusion model starting from pure noise.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :] * 1000.0  # scale up - t in [0,1] is a narrow range
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class FiLMDoubleConv(nn.Module):
    """Same as model.py's DoubleConv, but FiLM-modulated by a shared time embedding
    after the first conv/norm/act - each block gets its own small Linear
    projecting the shared emb_dim down to this block's (scale, shift) pair."""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.film = nn.Linear(emb_dim, 2 * out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x, emb):
        h = self.act1(self.norm1(self.conv1(x)))
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.act2(self.norm2(self.conv2(h)))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = FiLMDoubleConv(in_ch, out_ch, emb_dim)

    def forward(self, x, emb):
        return self.conv(self.pool(x), emb)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = FiLMDoubleConv(in_ch // 2 + skip_ch, out_ch, emb_dim)

    def forward(self, x, skip, emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, emb)


class FlowUNet(nn.Module):
    """Predicts velocity v_theta(x_t, t). Same 4-level topology as model.UNet."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, time_emb_dim: int = 128):
        super().__init__()
        c = base_channels
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        emb_dim = time_emb_dim * 4

        self.stem = FiLMDoubleConv(in_channels, c, emb_dim)
        self.down1 = Down(c, c * 2, emb_dim)
        self.down2 = Down(c * 2, c * 4, emb_dim)
        self.down3 = Down(c * 4, c * 8, emb_dim)
        self.bottleneck = Down(c * 8, c * 16, emb_dim)

        self.up3 = Up(c * 16, c * 8, c * 8, emb_dim)
        self.up2 = Up(c * 8, c * 4, c * 4, emb_dim)
        self.up1 = Up(c * 4, c * 2, c * 2, emb_dim)
        self.up0 = Up(c * 2, c, c, emb_dim)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)
        # zero-init: model starts by predicting v=0 (i.e. "stay at x_0"),
        # a stable starting point to learn the correction from
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t)

        x0 = self.stem(x, emb)
        x1 = self.down1(x0, emb)
        x2 = self.down2(x1, emb)
        x3 = self.down3(x2, emb)
        xb = self.bottleneck(x3, emb)

        y = self.up3(xb, x3, emb)
        y = self.up2(y, x2, emb)
        y = self.up1(y, x1, emb)
        y = self.up0(y, x0, emb)

        return self.head(y)


@torch.no_grad()
def sample_ode(model: FlowUNet, x0: torch.Tensor, n_steps: int = 4) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x, t) from t=0 (x0 = low_log) to t=1 (predicted
    high_log), simple Euler steps. x0 is close to x1 by construction (same
    large-scale structure), so few steps should suffice - unlike diffusion
    sampling from pure noise, which typically needs dozens+."""
    model.eval()
    x = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        v = model(x, t)
        x = x + v * dt
    return x
