"""Deterministic residual-correction UNet on HEALPix patches (2D images).

Not generative: instead of a flow, this learns a DIFFERENCE map and adds it to the
DISCO input. Per HEALPix NESTED patch (reshaped to an (L,L) image via the Morton map):

    input  = s_disco                         (arcsinh-signal of the DISCO patch)
    target = diff = s_high - s_disco         (preprocessing difference, per shell)
    pred   = DiffUNet(s_disco | cosmo, z)
    corrected = s_disco + pred               (== s_high when pred is perfect)
    loss   = mean( (corrected - s_high)^2 )  = mean( (pred - diff)^2 )   [pixel MSE]

Architecture (as requested): the encoder downsamples the patch while WIDENING the
channels up to `wide` (=512). A bottleneck then COMPRESSES the channels 512 -> ... -> 64
in steps, ADDS the (cosmology, redshift) embedding at the 64-wide latent, and EXPANDS
64 -> ... -> 512. The decoder upsamples back to the patch with skip connections and
emits the 1-channel difference. Conditioning enters ONLY at the 64-wide latent.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# HEALPix NESTED patch <-> 2D image  (Morton / Z-order de-interleave)
# ---------------------------------------------------------------------------
def nested_grid_perms(L: int):
    """Index arrays mapping a HEALPix NESTED patch (L*L pixels) to/from an (L,L) image.

        image = patch[to_img].reshape(L, L)
        patch = image.reshape(L * L)[to_patch]

    Within one HEALPix base-resolution pixel the NESTED local index is a Z-order
    (Morton) curve: even bits index one intra-face axis, odd bits the other, so
    de-interleaving the bits recovers the true 2D (x, y) coordinates. L is a power of 2.
    """
    L2 = L * L
    nest = np.arange(L2, dtype=np.int64)
    x = np.zeros(L2, np.int64)
    y = np.zeros(L2, np.int64)
    b = 0
    while (1 << (2 * b)) < L2:
        x |= ((nest >> (2 * b)) & 1) << b
        y |= ((nest >> (2 * b + 1)) & 1) << b
        b += 1
    pos = y * L + x
    to_img = np.empty(L2, np.int64)
    to_img[pos] = nest
    to_patch = pos
    return to_img, to_patch


def downscale_nested(maps: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downscale of NESTED-ordered HEALPix maps: nside -> nside/factor.

    NESTED indexing groups every parent pixel's factor^2 children CONTIGUOUSLY, so
    downscaling is just reshape + mean -- exactly equivalent to
    healpy.ud_grade(..., order_in='NESTED', order_out='NESTED') but pure numpy (no
    healpy call per shell) and fast. Used for quick, low-resolution dev iterations
    (smaller patches -> far fewer FLOPs) before committing to a full-res run.
    maps: (..., npix) -> (..., npix // factor**2). factor must be a power of 2.
    """
    if factor <= 1:
        return maps
    block = factor * factor
    *lead, npix = maps.shape
    assert npix % block == 0, f"npix={npix} not divisible by factor^2={block}"
    return maps.reshape(*lead, npix // block, block).mean(axis=-1)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
def _cgs(cin, cout, groups):
    """GroupNorm -> SiLU -> 3x3 Conv."""
    return nn.Sequential(nn.GroupNorm(min(groups, cin), cin), nn.SiLU(),
                         nn.Conv2d(cin, cout, 3, padding=1))


class ResBlock(nn.Module):
    """Two GN-SiLU-Conv layers + residual skip (no conditioning: cond enters only
    at the bottleneck latent)."""

    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.body = nn.Sequential(_cgs(cin, cout, groups), _cgs(cout, cout, groups))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class DiffUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, ch_mult=(1, 2, 4, 8),
                 bottleneck=64, cond_dim=17, groups=8):
        super().__init__()
        chs = [base * m for m in ch_mult]          # e.g. [64, 128, 256, 512]
        wide = chs[-1]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # encoder: ResBlock (save skip) then strided-conv downsample, widening channels
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for c in chs:
            self.enc_blocks.append(ResBlock(prev, c, groups))
            self.downs.append(nn.Conv2d(c, c, 4, stride=2, padding=1))
            prev = c                                # prev == wide after the loop

        # bottleneck: compress wide -> ... -> bottleneck (in steps), inject (cosmo, z),
        # then expand bottleneck -> ... -> wide.
        comp, steps, c = [], [], wide
        while c > bottleneck:
            steps.append(c)
            comp.append(_cgs(c, c // 2, groups)); c //= 2
        self.compress = nn.Sequential(*comp)        # wide -> bottleneck
        self.bottleneck = c
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, self.bottleneck), nn.SiLU(),
                                      nn.Linear(self.bottleneck, self.bottleneck))
        self.expand = nn.Sequential(*[_cgs(c // 2, c, groups) for c in reversed(steps)])

        # decoder: transpose-conv upsample, concat skip, ResBlock
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev = wide
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 4, stride=2, padding=1))
            self.dec_blocks.append(ResBlock(c + c, c, groups))
            prev = c

        self.out = nn.Sequential(nn.GroupNorm(min(groups, prev), prev), nn.SiLU(),
                                 nn.Conv2d(prev, out_ch, 3, padding=1))

    def forward(self, x, cosmo):
        # x: (B, 1, L, L) DISCO signal patch; cosmo: (B, cond_dim) = [cosmo params, z]
        if x.dim() == 3:
            x = x[:, None]
        h = self.in_conv(x)
        skips = []
        for blk, dn in zip(self.enc_blocks, self.downs):
            h = blk(h)
            skips.append(h)
            h = dn(h)
        h = self.compress(h)                              # -> 64-wide latent
        h = h + self.cond_mlp(cosmo)[:, :, None, None]    # add (cosmo, z) at the latent
        h = self.expand(h)                                # -> 512
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)
            h = blk(torch.cat([h, skips.pop()], 1))
        return self.out(h)                                # predicted difference (B,1,L,L)


def correction_loss(pred_diff, target_diff, huber_delta: float = 0.1):
    """Robust pixel loss: Huber(corrected, high) = Huber(pred_diff, high - disco).

    Plain MSE let rare bright/outlier pixels (cluster spikes, shot-noise peaks) dominate
    a batch's loss (observed: per-batch loss jumping between ~1e-4 and >5 while the
    median batch was near 0) -- pure batch-to-batch content variance, not divergence,
    but it drowns out the learning signal from typical batches. Huber is quadratic
    (== MSE) for small errors and linear beyond huber_delta, so those rare outliers
    stop dominating the gradient while ordinary batches train the same as before.
    """
    return nn.functional.smooth_l1_loss(pred_diff, target_diff, beta=huber_delta)


# ---------------------------------------------------------------------------
# Spectral (radial power spectrum) loss
# ---------------------------------------------------------------------------
# Plain pixel MSE is minimized by the CONDITIONAL MEAN of high | disco, cosmo, z.
# Wherever the true small-scale field has stochastic structure not fully determined
# by the input (shot noise, sub-patch detail), the conditional mean is a SMOOTHED,
# lower-variance map: MSE regression systematically suppresses power at every scale
# (confirmed empirically: corrected Cl/high ratio sat at 0.5-0.95 across ALL ell, not
# just high-ell detail). This term matches the 2D FFT radial power spectrum of the
# corrected patch to the truth's, directly penalizing that power loss.
def radial_bins(L: int, device, n_bins: int = None):
    """Precompute the (bin index per rFFT frequency, bin counts) for an (L,L) patch."""
    n_bins = n_bins or L // 2
    fy = torch.fft.fftfreq(L, device=device) * L          # (L,) in cycles/patch
    fx = torch.fft.rfftfreq(L, device=device) * L          # (L//2+1,)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)    # (L, L//2+1)
    bins = torch.clamp((r / r.max() * (n_bins - 1)).long(), 0, n_bins - 1).view(-1)
    counts = torch.bincount(bins, minlength=n_bins).clamp(min=1)
    return bins, counts, n_bins


def radial_power(img: torch.Tensor, bins: torch.Tensor, counts: torch.Tensor,
                 n_bins: int) -> torch.Tensor:
    """(B,1,L,L) -> (B, n_bins) mean 2D-FFT power per radial bin (the patch's own Cl analogue)."""
    B = img.shape[0]
    f = torch.fft.rfft2(img.squeeze(1))                    # (B, L, L//2+1) complex
    power = (f.real ** 2 + f.imag ** 2).view(B, -1)        # (B, L*(L//2+1))
    binned = torch.zeros(B, n_bins, device=img.device, dtype=power.dtype)
    binned.scatter_add_(1, bins.unsqueeze(0).expand(B, -1), power)
    return binned / counts.unsqueeze(0)


def spectral_loss(corrected_img: torch.Tensor, target_img: torch.Tensor,
                  bins: torch.Tensor, counts: torch.Tensor, n_bins: int,
                  eps: float = 1e-3) -> torch.Tensor:
    """Mean squared LOG-RATIO of radial power spectra: mean( (log((Pc+eps)/(Pt+eps)))^2 ).

    Earlier version used (log1p(Pc)-log1p(Pt))^2, which behaves like an ABSOLUTE
    difference for the small power values at high ell/bin index (log1p(x)~x for
    x<<1) -- so gradients were dominated by the much larger low-ell power terms, and
    empirically the model learned to correct only large scales, leaving the
    corrected/high Cl ratio at high ell glued to the (uncorrected) DISCO/high ratio.
    The log-RATIO form is scale-invariant per bin: a given FRACTIONAL power error
    contributes the same loss regardless of whether it's a high- or low-power bin, so
    small-scale (high-ell) accuracy is no longer drowned out. eps is negligible
    relative to typical per-bin power (>>1e-3 in these units); it only guards zero bins.
    """
    pc = radial_power(corrected_img, bins, counts, n_bins)
    pt = radial_power(target_img, bins, counts, n_bins)
    return torch.log((pc + eps) / (pt + eps)).pow(2).mean()
