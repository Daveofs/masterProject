"""Time-conditioned UNet for conditional flow matching.

Unlike model.py's UNet (deterministic, low -> high directly), this predicts a
velocity field v_theta(x_t, t) used to integrate an ODE from x_0 = low_f
(the low-fidelity patch, in whichever --space dataset.transform_pair selects --
'delta' [linear overdensity] by default since 2026-07-18, 'log1p' only for
pre-2026-07-18 checkpoints -- NOT noise) to x_1, along the
straight-line interpolant x_t = (1-t)*x_0 + t*x_1.

Starting the flow from the low-fidelity map instead of Gaussian noise is the
whole point here: x_0 is already close to x_1 (same large-scale structure,
only small-scale correction differs), so the ODE trajectory is close to
straight and should need very few integration steps at inference - unlike a
typical diffusion model starting from pure noise.

HIGH-PASS RESIDUAL formulation (2026-07-21, ported from diffusion/model.py after
comparing all three pipelines' cl_ratio_by_zbin_grid.png / kappa_cl_pctile_band.png:
diffusion's kappa Cl stayed near 1.0 everywhere; this model's and sphereflow's both
showed a several-percent systematic LOW bias at every ell, and this model additionally
showed a catastrophic percentile-band collapse on the sparsest shells -- see
[[deepsphere-shell-correction]] memory). Root cause: x_1 used to be high_f directly,
so the ODE was free to drift x_0's LARGE scales too, even though DISCO's large scales
are already correct and need no correction -- any per-shell miscalibration there
compounds coherently across the dozens of shells a kappa map integrates, exactly
matching the observed kappa failure. Fix: x_1 is now x_0 + residual_target(x_0,
high_f) -- a target that is IDENTICAL to x_0 at large scales (below --hp-cutoff)
and only diverges from it at small scales -- so the target velocity x_1-x_0 is a pure
high-pass field, and integrating it can only ever add small-scale content. See
residual_target/compose_corrected below and diffusion/model.py's docstring for the
full mechanism (identical math, ported here since this is a genuinely different
architecture -- see feedback-decoupled-pipeline-modules memory, no cross-import).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# High-pass residual machinery -- IDENTICAL math to diffusion/model.py's (ported,
# not imported: this is a from-scratch FlowUNet, not the EDM denoiser, but the
# "pin large scales to the low map" mechanism is architecture-agnostic). Free
# functions so train_flow.py and apply_flow.py both call the SAME code to build the
# target and to compose the corrected map, and therefore cannot silently drift.
# ---------------------------------------------------------------------------

_HP_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _highpass_mask(H: int, W: int, cutoff_frac: float, transition_frac: float,
                   device, dtype) -> torch.Tensor:
    """Real radial high-pass mask on the rfft2 grid of an (H, W) patch. 0 below
    cutoff_frac, smoothly (raised-cosine) rising to 1 by cutoff_frac+transition_frac,
    both fractions of the patch Nyquist radial frequency."""
    key = (H, W, round(cutoff_frac, 5), round(transition_frac, 5), device, dtype)
    m = _HP_MASK_CACHE.get(key)
    if m is not None:
        return m
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.rfftfreq(W, device=device)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5
    lo, hi = cutoff_frac, cutoff_frac + transition_frac
    t = torch.clamp((r - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    mask = (0.5 * (1 - torch.cos(math.pi * t))).to(dtype)
    _HP_MASK_CACHE[key] = mask
    return mask


def highpass_2d(x: torch.Tensor, cutoff_frac: float, transition_frac: float) -> torch.Tensor:
    """Radial high-pass filter on (B, C, H, W) patches via 2D rFFT -- removes the
    largest-scale content, keeping only small scales. See diffusion/model.py's
    identical function for the full rationale."""
    B, C, H, W = x.shape
    mask = _highpass_mask(H, W, cutoff_frac, transition_frac, x.device, x.dtype)
    f = torch.fft.rfft2(x) * mask[None, None]
    return torch.fft.irfft2(f, s=(H, W))


def residual_target(low_f: torch.Tensor, high_f: torch.Tensor,
                    cutoff_frac: float = 0.10, transition_frac: float = 0.10) -> torch.Tensor:
    """The flow TARGET's high-pass component: highpass(high_f - low_f). Combined
    with x0=low_f, x1 = x0 + residual_target(...) makes the target velocity
    x1-x0 == this residual exactly -- a pure high-pass field, so the ODE can only
    ever add small-scale content (large scales start AND end at low_f)."""
    return highpass_2d(high_f - low_f, cutoff_frac, transition_frac)


def compose_corrected(low_f: torch.Tensor, sample: torch.Tensor,
                      cutoff_frac: float = 0.10, transition_frac: float = 0.10) -> torch.Tensor:
    """Re-assemble the corrected field from whatever the ODE produced: low_f +
    highpass(sample). `sample` here is (ODE output - low_f), i.e. the EFFECTIVE
    residual the integration actually accumulated -- highpass-ing it AGAIN before
    adding back is a hard guarantee that large scales end up EXACTLY low_f's,
    independent of any large-scale drift finite-step Euler integration left in
    (idempotent for an already-high-pass field, so this is a no-op when the ODE
    behaved perfectly). THE inverse of residual_target -- train and apply must both
    go through this pair, see diffusion/model.py's identical compose_corrected."""
    return low_f + highpass_2d(sample, cutoff_frac, transition_frac)


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
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None] # rescales and shifts every channel content
        return self.act2(self.norm2(self.conv2(h)))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2) # Half the spatial resolution
        self.conv = FiLMDoubleConv(in_ch, out_ch, emb_dim) # Doubles the channels

    def forward(self, x, emb):
        return self.conv(self.pool(x), emb)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2) # Double the spatial resolution
        self.conv = FiLMDoubleConv(in_ch // 2 + skip_ch, out_ch, emb_dim)  # Halves the channels

    def forward(self, x, skip, emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, emb)


class FlowUNet(nn.Module):
    """Predicts velocity v_theta(x_t, t [, cosmo_z]). Same 4-level topology as model.UNet.

    cosmo_z conditioning (H0->h, Omega_cdm, Ob, Om, ns, s8, w0, redshift -- see
    dataset.cosmo_z_vector) is injected ONCE, additively, into the bottleneck latent
    (between self.bottleneck and self.up3) -- not threaded through every FiLM block
    like the time embedding, since that's the one place requested and it keeps the
    change small/easy to ablate. use_cosmo_cond defaults to True (added by default);
    set it False to reproduce the original, unconditioned model for comparison -- the
    two are architecturally identical everywhere except this one addition.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, time_emb_dim: int = 128,
                 use_cosmo_cond: bool = True, cosmo_z_dim: int = 8):
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

        self.use_cosmo_cond = use_cosmo_cond
        if use_cosmo_cond:
            bottleneck_ch = c * 16
            self.cosmo_mlp = nn.Sequential(
                nn.Linear(cosmo_z_dim, bottleneck_ch), nn.SiLU(inplace=True),
                nn.Linear(bottleneck_ch, bottleneck_ch),
            )
            # zero-init: this addition starts as v=0 contribution (a no-op, identical
            # to use_cosmo_cond=False at init), same stability rationale as the head.
            nn.init.zeros_(self.cosmo_mlp[-1].weight)
            nn.init.zeros_(self.cosmo_mlp[-1].bias)

        self.up3 = Up(c * 16, c * 8, c * 8, emb_dim)
        self.up2 = Up(c * 8, c * 4, c * 4, emb_dim)
        self.up1 = Up(c * 4, c * 2, c * 2, emb_dim)
        self.up0 = Up(c * 2, c, c, emb_dim)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)
        # zero-init: model starts by predicting v=0 (i.e. "stay at x_0"),
        # a stable starting point to learn the correction from
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
               cosmo_z: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.time_embed(t)

        x0 = self.stem(x, emb)
        x1 = self.down1(x0, emb)
        x2 = self.down2(x1, emb)
        x3 = self.down3(x2, emb)
        xb = self.bottleneck(x3, emb)

        if self.use_cosmo_cond:
            if cosmo_z is None:
                raise ValueError("this FlowUNet was built with use_cosmo_cond=True: "
                                 "forward() requires cosmo_z (see dataset.cosmo_z_vector)")
            xb = xb + self.cosmo_mlp(cosmo_z)[:, :, None, None]

        y = self.up3(xb, x3, emb)
        y = self.up2(y, x2, emb)
        y = self.up1(y, x1, emb)
        y = self.up0(y, x0, emb)

        return self.head(y)


@torch.no_grad()
def sample_ode(model: FlowUNet, x0: torch.Tensor, n_steps: int = 4,
              cosmo_z: torch.Tensor | None = None, amp: bool = False) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x, t [, cosmo_z]) from t=0 (x0 = low_f) to t=1
    (predicted high_f), simple Euler steps. x0 is close to x1 by construction (same
    large-scale structure), so few steps should suffice - unlike diffusion
    sampling from pure noise, which typically needs dozens+.

    amp=True runs each model() call under bf16 autocast -- same pattern and
    rationale as sphere_flow.sample_ode's own amp flag (memory-bandwidth-bound
    2D convs benefit the same way as the graph gather-convs do on this hardware).
    The Euler accumulator x stays fp32: model(...) returns bf16 under autocast,
    but `x (fp32) + bf16_tensor * dt` type-promotes to fp32 automatically, so
    precision doesn't degrade across steps. Default False here (unlike
    sphere_flow's default True) because this was never benchmarked for unet's
    FlowUNet -- opt in via --amp once measured on a real reconstruction."""
    model.eval()
    x = x0.clone()
    dt = 1.0 / n_steps
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            v = model(x, t, cosmo_z=cosmo_z)
            x = x + v * dt
    return x
