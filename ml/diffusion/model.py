"""EDM-style conditional diffusion (Karras, Aittala, Aila & Laine 2022, "Elucidating
the Design Space of Diffusion-Based Generative Models") for the low->high patch
correction.

Unlike unet/flow_model.py's FlowUNet (ODE starts at x0 = low_log, needs only ~4-8
Euler steps because the straight-line interpolant is close to a no-op) and
sphereflow/sphere_flow.py (rectified flow from x0 ~ N(0,I), also a straight-line
interpolant), this model draws EVERY sample from pure noise
x_T ~ N(0, sigma_max^2 I) *independent of the conditioning value* -- cond=low_log
only steers the denoiser at each step via channel-concat, never sets the starting
point -- and follows the actual EDM noise schedule + Heun 2nd-order sampler
(~32 steps by default), not a 2-point straight-line flow. This is the genuinely
different generative process asked for: a real multi-step diffusion model, not
another rectified flow. Slower to sample; the intended payoff is a less
constrained (non-straight-line) generative trajectory for the faint-shell /
high-ell undercorrection both flow models still show (see
sphereflow-model-survey memory).

Architecture building blocks (SinusoidalEmbedding, FiLMDoubleConv, Down, Up) are a
deliberate LOCAL duplicate of unet/flow_model.py's -- not an import, see
feedback-decoupled-pipeline-modules memory: each ml/ pipeline stays self-contained
so a rename in one never silently breaks another. Only two things differ from
FlowUNet: the stem takes 2 input channels (noisy x_t + cond, concatenated) instead
of 1, and the embedded scalar is EDM's c_noise(sigma) instead of a flow time t in
[0,1] (so the embedding does not need FlowUNet's *1000 rescale -- c_noise already
spans a useful range, see SinusoidalEmbedding).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Same construction as FlowUNet's SinusoidalTimeEmbedding, but no *1000 rescale:
    c_noise = log(sigma)/4 already spans O(1)-O(10) (sigma in [0.002, 80] -> c_noise
    in about [-1.55, 1.10] most of the mass, wider at the tails), unlike a flow's
    t in [0,1] which needs the rescale to avoid all frequencies aliasing together."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device).float() / half)
        args = x[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class FiLMDoubleConv(nn.Module):
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


class DenoiserUNet(nn.Module):
    """F_theta(concat(c_in*x_t, cond), c_noise(sigma) [, cosmo_z]) -- the raw network
    EDMPrecond wraps. Same 4-level FiLM U-Net topology as FlowUNet; in_channels=2
    (noisy target + conditioning map, channel-concatenated) is the only structural
    difference besides the embedding input. cosmo_z conditioning identical to
    FlowUNet's: injected once, additively, zero-initialized, at the bottleneck."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1,
                 base_channels: int = 32, emb_dim_mult: int = 4, noise_emb_dim: int = 128,
                 use_cosmo_cond: bool = True, cosmo_z_dim: int = 8):
        super().__init__()
        c = base_channels
        self.noise_embed = nn.Sequential(
            SinusoidalEmbedding(noise_emb_dim),
            nn.Linear(noise_emb_dim, noise_emb_dim * emb_dim_mult),
            nn.SiLU(inplace=True),
            nn.Linear(noise_emb_dim * emb_dim_mult, noise_emb_dim * emb_dim_mult),
        )
        emb_dim = noise_emb_dim * emb_dim_mult

        self.stem = FiLMDoubleConv(in_channels, c, emb_dim)
        self.down1 = Down(c, c * 2, emb_dim)
        self.down2 = Down(c * 2, c * 4, emb_dim)
        self.down3 = Down(c * 4, c * 8, emb_dim)
        self.bottleneck = Down(c * 8, c * 16, emb_dim)

        self.use_cosmo_cond = use_cosmo_cond
        if use_cosmo_cond:
            bottleneck_ch = c * 16
            self.cosmo_mlp = nn.Sequential(
                nn.Linear(cosmo_z_dim, bottleneck_ch), nn.SiLU(inplace=True),
                nn.Linear(bottleneck_ch, bottleneck_ch),
            )
            nn.init.zeros_(self.cosmo_mlp[-1].weight)
            nn.init.zeros_(self.cosmo_mlp[-1].bias)

        self.up3 = Up(c * 16, c * 8, c * 8, emb_dim)
        self.up2 = Up(c * 8, c * 4, c * 4, emb_dim)
        self.up1 = Up(c * 4, c * 2, c * 2, emb_dim)
        self.up0 = Up(c * 2, c, c, emb_dim)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)
        # zero-init: F_theta starts at 0, so D_x = c_skip*x_t + 0 at init -- a stable,
        # well-defined starting point for EDM preconditioning (same rationale as
        # FlowUNet's zero-init head, adapted to the denoiser's own identity).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor,
               cosmo_z: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.noise_embed(c_noise)

        x0 = self.stem(x, emb)
        x1 = self.down1(x0, emb)
        x2 = self.down2(x1, emb)
        x3 = self.down3(x2, emb)
        xb = self.bottleneck(x3, emb)

        if self.use_cosmo_cond:
            if cosmo_z is None:
                raise ValueError("this DenoiserUNet was built with use_cosmo_cond=True: "
                                 "forward() requires cosmo_z (see dataset.cosmo_z_vector)")
            xb = xb + self.cosmo_mlp(cosmo_z)[:, :, None, None]

        y = self.up3(xb, x3, emb)
        y = self.up2(y, x2, emb)
        y = self.up1(y, x1, emb)
        y = self.up0(y, x0, emb)

        return self.head(y)


class EDMPrecond(nn.Module):
    """Karras et al. 2022 preconditioning, adapted for image-to-image conditioning:
    cond is channel-concatenated to the (c_in-scaled) noisy input -- cond itself is
    never noised or rescaled by c_in, since it isn't part of the diffusion process,
    only steers it (same convention as e.g. Palette/SR3-style conditional diffusion).

        c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out(sigma)  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        c_in(sigma)   = 1 / sqrt(sigma^2 + sigma_data^2)
        c_noise(sigma)= log(sigma) / 4
        D_theta(x, sigma, cond) = c_skip*x + c_out * F_theta(cat(c_in*x, cond), c_noise)

    sigma_data should match the actual std of the target field (high_log) -- see
    train_diffusion.py's estimate_sigma_data, which measures it from real data
    instead of using EDM's CIFAR-tuned default of 0.5 blindly."""

    def __init__(self, net: DenoiserUNet, sigma_data: float = 0.5):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor,
               cosmo_z: torch.Tensor | None = None) -> torch.Tensor:
        sd = self.sigma_data
        sigma = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
        c_noise = (0.25 * torch.log(sigma)).flatten()

        net_in = torch.cat([c_in * x, cond], dim=1)
        F_x = self.net(net_in, c_noise, cosmo_z=cosmo_z)
        return c_skip * x + c_out * F_x


def edm_loss(precond: EDMPrecond, x1: torch.Tensor, cond: torch.Tensor, sigma_data: float,
            cosmo_z: torch.Tensor | None = None, p_mean: float = -1.2,
            p_std: float = 1.2) -> torch.Tensor:
    """EDM training loss: sample ln(sigma) ~ N(p_mean, p_std^2) per example, noise
    x1 to x_t = x1 + sigma*eps, and weight the denoising MSE by
    (sigma^2+sigma_data^2)/(sigma*sigma_data)^2 so every noise level contributes
    comparable gradient magnitude (Karras et al. eq. 8).

    sigma_data is taken as an explicit argument rather than read off
    `precond.sigma_data` because the caller may pass a DDP-wrapped precond -- DDP
    only forwards `.forward()`, not arbitrary attributes, so `precond.sigma_data`
    raises AttributeError under `torch.nn.parallel.DistributedDataParallel`. The
    caller already knows sigma_data (it's what EDMPrecond was constructed with)."""
    bs = x1.shape[0]
    rnd = torch.randn(bs, device=x1.device, dtype=torch.float32)
    sigma = torch.exp(p_mean + p_std * rnd).to(x1.dtype).reshape(-1, 1, 1, 1)
    sd = sigma_data
    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2

    noise = torch.randn_like(x1) * sigma
    x_t = x1 + noise
    d_x = precond(x_t, sigma.flatten(), cond, cosmo_z=cosmo_z)
    return (weight * (d_x - x1) ** 2).mean()


@torch.no_grad()
def sample_heun(precond: EDMPrecond, cond: torch.Tensor, n_steps: int = 32,
                cosmo_z: torch.Tensor | None = None, sigma_min: float = 0.002,
                sigma_max: float = 80.0, rho: float = 7.0) -> torch.Tensor:
    """EDM Algorithm 1, deterministic (S_churn=0): 2nd-order Heun ODE solver over the
    Karras sigma schedule, starting from x_T ~ N(0, sigma_max^2 I) -- independent of
    cond, unlike sample_ode's x0=low_log. n_steps=32 by default (vs. the flow
    models' 4-8): a real diffusion trajectory is not a straight line, so it needs
    materially more function evaluations -- that is the whole point of trying this
    alternative, not a bug to tune away."""
    precond.eval()
    device, dtype = cond.device, cond.dtype
    bs = cond.shape[0]

    step = torch.arange(n_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step / max(n_steps - 1, 1) *
              (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, dtype=torch.float64, device=device)])  # sigma_N = 0

    x = torch.randn(cond.shape, device=device, dtype=dtype) * t_steps[0].to(dtype)

    for i in range(n_steps):
        sigma_cur = t_steps[i].to(dtype).expand(bs)
        sigma_next = t_steps[i + 1].to(dtype).expand(bs)

        d_cur = (x - precond(x, sigma_cur, cond, cosmo_z=cosmo_z)) / sigma_cur.reshape(-1, 1, 1, 1)
        x_next = x + (sigma_next - sigma_cur).reshape(-1, 1, 1, 1) * d_cur

        if t_steps[i + 1] > 0:
            d_next = (x_next - precond(x_next, sigma_next, cond, cosmo_z=cosmo_z)) / sigma_next.reshape(-1, 1, 1, 1)
            x_next = x + (sigma_next - sigma_cur).reshape(-1, 1, 1, 1) * 0.5 * (d_cur + d_next)

        x = x_next

    return x
