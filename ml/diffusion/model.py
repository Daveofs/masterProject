"""EDM-style conditional diffusion (Karras, Aittala, Aila & Laine 2022, "Elucidating
the Design Space of Diffusion-Based Generative Models") for the low->high patch
correction -- **HIGH-PASS RESIDUAL formulation**.

WHAT THE MODEL DIFFUSES (this is the key design decision, changed 2026-07-17 after
the first full-field run failed -- see diffusion-pipeline-build memory):
The model does NOT generate the whole high-res field from noise. It generates only
the small-scale (high-pass) RESIDUAL between high and low:

    target x1 = highpass(high_f - low_f)          (small scales only)
    cond      = low_f                                 (all scales, channel-concat)
    corrected = low_f + highpass(diffusion_sample)   (compose at inference)

low_f/high_f are whatever field --space selects (train_diffusion.py's transform_pair):
'delta' (linear overdensity, the space analysis.full_sky.od_cl actually measures) by
default since 2026-07-18, 'log1p' kept only for pre-2026-07-18 checkpoints/comparison
-- see dataset.raw_to_delta_pair's docstring for the measured reason log1p was a
formulation bug, not a detail.

so the LARGE scales of the output are pinned EXACTLY to the low (DISCO) map and are
never touched by the generator. This is dictated by the physics + the failure of the
first (full-field) run:
  - Physics (see transfer-fn / transfer_function.py): DISCO's large scales are
    already correct (transfer function T~1, phase corr r~1 at large scales); only
    the small-scale amplitude is wrong. So there is nothing to gain by regenerating
    large scales, and everything to lose.
  - Failure mode of full-field-from-noise: each patch drew its large-scale content
    from INDEPENDENT noise, so per-patch power was ~right but the large-scale PHASES
    disagreed between neighbouring patches. After overlap-tile stitching, the
    cross-patch large-scale modes averaged incoherently -> the full-sky / weak-lensing
    kappa Cl at low ell was destroyed (wild 0.4-1.6 scatter), even though the
    per-patch 2D-FFT power looked fine. Pinning large scales to the (globally
    coherent) low map fixes this BY CONSTRUCTION.
The flow models (unet/sphereflow) avoid the same trap differently: unet starts its
ODE from x0=low so large scales are inherited; here we instead let the process start
from pure noise (a genuine multi-step EDM diffusion, the point of this pipeline) but
CONSTRAIN it to the high-pass band via highpass() on both the target and the sample.

Everything ELSE is standard EDM: draw every sample from x_T ~ N(0, sigma_max^2 I),
follow the Karras noise schedule + Heun 2nd-order sampler (~32 steps by default). The
denoiser still conditions on the FULL low_f (all scales) via channel-concat -- it
needs the large-scale context to know which small-scale structure to synthesize
(low and high share small-scale PHASES per the transfer-fn memo), even though its
OUTPUT is high-pass.

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


# ---------------------------------------------------------------------------
# High-pass residual machinery (the formulation-defining part -- see module
# docstring). Kept as free functions so train_diffusion.py and apply_diffusion.py
# both call the SAME code to build the target and to compose the corrected map,
# and therefore cannot silently drift (a mismatch between how training filters the
# target and how inference re-composes it would corrupt every result).
# ---------------------------------------------------------------------------

# (H, W, cutoff, transition, device, dtype) -> radial high-pass mask over the rFFT
# grid. Depends only on geometry + cutoff, not on the data, so it is built once per
# distinct patch size and reused for every patch/batch.
_HP_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _highpass_mask(H: int, W: int, cutoff_frac: float, transition_frac: float,
                   device, dtype) -> torch.Tensor:
    """Real radial high-pass mask on the rfft2 grid of an (H, W) patch. 0 below
    cutoff_frac, smoothly (raised-cosine) rising to 1 by cutoff_frac+transition_frac,
    where both fractions are of the patch NYQUIST radial frequency. A smooth (not
    hard) transition avoids the ringing a brick-wall cutoff would inject into the
    small-scale residual."""
    key = (H, W, round(cutoff_frac, 5), round(transition_frac, 5), device, dtype)
    m = _HP_MASK_CACHE.get(key)
    if m is not None:
        return m
    fy = torch.fft.fftfreq(H, device=device)      # cycles/pixel in [-0.5, 0.5)
    fx = torch.fft.rfftfreq(W, device=device)      # cycles/pixel in [0, 0.5]
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5   # 1.0 == Nyquist
    lo, hi = cutoff_frac, cutoff_frac + transition_frac
    t = torch.clamp((r - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    mask = (0.5 * (1 - torch.cos(math.pi * t))).to(dtype)       # 0 below lo, 1 above hi
    _HP_MASK_CACHE[key] = mask
    return mask


def highpass_2d(x: torch.Tensor, cutoff_frac: float, transition_frac: float) -> torch.Tensor:
    """Radial high-pass filter on (B, C, H, W) patches via 2D rFFT: removes the
    largest-scale (low radial-wavenumber) content, keeping only small scales.
    cutoff_frac / transition_frac are fractions of the patch Nyquist frequency
    (e.g. cutoff 0.1 pins everything below ~0.1*Nyquist to whatever the caller adds
    it back onto). Operates per patch, so it says nothing about cross-patch coherence
    on its own -- that comes from adding the result onto the (globally coherent) low
    map in compose_corrected."""
    B, C, H, W = x.shape
    mask = _highpass_mask(H, W, cutoff_frac, transition_frac, x.device, x.dtype)
    f = torch.fft.rfft2(x) * mask[None, None]
    return torch.fft.irfft2(f, s=(H, W))


def residual_target(low_f: torch.Tensor, high_f: torch.Tensor,
                    cutoff_frac: float, transition_frac: float) -> torch.Tensor:
    """The diffusion TARGET x1: the high-pass part of the (high - low) residual. This
    is what the denoiser learns to produce -- small scales only, large scales dropped
    (they're supplied by the low map at compose time)."""
    return highpass_2d(high_f - low_f, cutoff_frac, transition_frac)


def compose_corrected(low_f: torch.Tensor, sample: torch.Tensor,
                      cutoff_frac: float, transition_frac: float) -> torch.Tensor:
    """Re-assemble the corrected field from a diffusion SAMPLE: low map (all its
    scales) + the high-pass sample. highpass() is applied to the sample again here
    (idempotent for an already-high-pass field) as a hard guarantee that the sample
    contributes NOTHING at large scales, so corrected's large scales == low's large
    scales exactly, independent of any low-frequency leakage the finite-step sampler
    might have left in. THE inverse of residual_target -- train and apply must both go
    through this pair."""
    return low_f + highpass_2d(sample, cutoff_frac, transition_frac)


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
        h = self.act1(self.norm1(self.conv1(x))) # (1) conv1: spatial change of features
        scale, shift = self.film(emb).chunk(2, dim=1) # (2) from σ
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None] # (3) FiLM: per-channel content (feature maps) rescale/offset
        return self.act2(self.norm2(self.conv2(h))) # (4) conv2: another spatial change


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

    sigma_data should match the actual std of the target field (high_f) -- see
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
    caller already knows sigma_data (it's what EDMPrecond was constructed with).

    PER-PATCH VARIANCE NORMALIZATION (2026-07-21, on top of EDM's own sigma-level
    `weight` above -- the two address different sources of scale variation): a
    dense/well-resolved shell's TRUE high-pass residual (x1) is naturally tiny, so
    without this, plain per-noise-level-weighted MSE still gives such patches far
    weaker gradient signal than a sparse shell's large residual -- the denoiser
    never gets pushed to calibrate "the right answer is near-zero here" precisely,
    showing up as a persistent few-percent under/overshoot in the densest-shell
    Cl-ratio panel that cutoff/transition tuning and more epochs both failed to fix
    (see [[deepsphere-shell-correction]] memory). Dividing by each patch's own x1
    variance makes every patch contribute comparably in RELATIVE terms regardless
    of how much correction that shell actually needs.

    BOUNDED, BATCH-RELATIVE (2026-07-21, fixed same day after job 4256969 showed
    the loss oscillating 40-70 for all 200 epochs with no convergence trend): a
    raw `1/(patch_var + eps)` divisor is UNBOUNDED for any patch whose true
    residual variance happens to be near-zero, and that compounds MULTIPLICATIVELY
    with EDM's own `weight` (which already has real dynamic range by design) --
    together they occasionally produce huge, noisy loss/gradient spikes that
    prevented convergence. Normalizing patch_var by the BATCH's own mean (so a
    "typical" patch gets a reweighting factor of 1) and clamping the result to
    >=0.1 (so no patch can be upweighted by more than 10x) keeps the SAME
    direction of correction -- low-variance patches still get relatively more
    weight -- without the unbounded blowup."""
    bs = x1.shape[0]
    rnd = torch.randn(bs, device=x1.device, dtype=torch.float32)
    sigma = torch.exp(p_mean + p_std * rnd).to(x1.dtype).reshape(-1, 1, 1, 1)
    sd = sigma_data
    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2

    noise = torch.randn_like(x1) * sigma
    x_t = x1 + noise
    d_x = precond(x_t, sigma.flatten(), cond, cosmo_z=cosmo_z)
    patch_var = x1.var(dim=(1, 2, 3), keepdim=True)
    rel_var = torch.clamp(patch_var / (patch_var.mean() + 1e-8), min=0.1)
    return (weight * (d_x - x1) ** 2 / rel_var).mean()


@torch.no_grad()
def sample_heun(precond: EDMPrecond, cond: torch.Tensor, n_steps: int = 32,
                cosmo_z: torch.Tensor | None = None, sigma_min: float = 0.002,
                sigma_max: float = 80.0, rho: float = 7.0,
                noise: torch.Tensor | None = None, amp: bool = False) -> torch.Tensor:
    """EDM Algorithm 1, deterministic (S_churn=0): 2nd-order Heun ODE solver over the
    Karras sigma schedule, starting from x_T ~ N(0, sigma_max^2 I) -- independent of
    cond, unlike sample_ode's x0=low_f. n_steps=32 by default (vs. the flow
    models' 4-8): a real diffusion trajectory is not a straight line, so it needs
    materially more function evaluations -- that is the whole point of trying this
    alternative, not a bug to tune away. Heun does UP TO 2 precond() calls per
    step, so this sampler is already the most expensive of the three pipelines'
    per-shell reconstruction cost; amp matters more here than anywhere else.

    noise: optional UNIT-variance N(0,I) tensor shaped like cond, used as the initial
    state (scaled by sigma_max here) instead of a fresh internal draw. This is what
    makes tiled full-sky reconstruction work: the solver is DETERMINISTIC given
    (noise, cond), so overlapping gnomonic tiles that are handed the same underlying
    sphere noise produce consistent output on their overlap and survive
    patch_tiling's averaging blend. With independent per-tile draws the blend
    destroys ~83% of the generated small-scale POWER (measured) -- see
    analysis/patch_tiling.tile_and_predict's pass_indices.

    amp=True runs each precond() call under bf16 autocast -- same pattern as
    sphere_flow.sample_ode's amp flag (flow_model.sample_ode mirrors it too).
    x/d_cur/d_next stay fp32 (precond returns bf16 under autocast, but
    fp32 +/- bf16 type-promotes to fp32), so precision doesn't degrade across
    steps. Default False here -- never benchmarked for this sampler; opt in via
    --amp once measured on a real reconstruction."""
    precond.eval()
    device, dtype = cond.device, cond.dtype
    bs = cond.shape[0]

    step = torch.arange(n_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step / max(n_steps - 1, 1) *
              (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, dtype=torch.float64, device=device)])  # sigma_N = 0

    if noise is None:
        noise = torch.randn(cond.shape, device=device, dtype=dtype)
    elif noise.shape != cond.shape:
        raise ValueError(f"sample_heun: noise shape {tuple(noise.shape)} != cond shape "
                         f"{tuple(cond.shape)}")
    x = noise.to(device=device, dtype=dtype) * t_steps[0].to(dtype)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
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
