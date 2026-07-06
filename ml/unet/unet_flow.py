"""Very simple 2D-UNet flow-matching generator on HEALPix patches.

A deliberately lightweight alternative to sphere_flow.py: instead of DeepSphere
graph convolutions, every HEALPix NESTED patch ((nside/order)^2 pixels) is reshaped
to an (L, L) image (via the intra-face Morton / Z-order map) and fed to a plain 2D
convolutional UNet. This is easy to read, fast on GPU, and scales trivially with DDP.

Conditional rectified flow matching (identical objective to sphere_flow):
    x_t = (1 - t) x0 + t x1,   x0 ~ N(0, I),   target velocity = x1 - x0,
learn v_theta(x_t, t | cond). Here cond = the DISCO (low-res) patch as an extra input
channel, plus the per-shell COSMOLOGY + REDSHIFT vector, which is embedded together
with the flow time t and ATTACHED TO THE LATENT features inside every ResBlock.

Sampling integrates dx/dt = v_theta from noise to a high-res patch (Euler).

The signal / overdensity / patch conventions are shared with sphere_flow.py (imported
in the trainer), so this model plugs into the exact same data loader.
"""

from __future__ import annotations
import math

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
    (Morton) curve: even bits index one intra-face axis, odd bits the other. So
    de-interleaving the bits recovers the true 2D (x, y) coordinates, which keeps
    spatially-adjacent pixels adjacent in the image (so the 2D convs are meaningful).
    Requires L to be a power of two (always true: L = nside/order, both powers of 2).
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
    pos = y * L + x                       # 2D flat position of nested pixel n
    to_img = np.empty(L2, np.int64)
    to_img[pos] = nest                    # image_flat[pos] = patch[nest]
    to_patch = pos                        # patch[nest] = image_flat[pos]
    return to_img, to_patch


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the scalar flow time t in [0, 1]."""
    if t.dim() == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=torch.float32)
                      / max(half - 1, 1))
    a = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(a), torch.cos(a)], -1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], -1)
    return emb


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """GN-SiLU-Conv twice + residual skip. The (time + cosmology + redshift)
    embedding is projected and ADDED to the latent feature map (FiLM-style shift),
    i.e. the conditioning is attached to the latent representation at every block."""

    def __init__(self, cin: int, cout: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(min(groups, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]        # attach cond to latent
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """Small conditional UNet velocity field v_theta(x_t, t | cond_map, cosmo).

    Input channels: [x_t, cond_map] (noisy target patch + DISCO patch). Output: the
    1-channel velocity. Cosmology+redshift and time are concatenated, embedded, and
    injected in every ResBlock. Number of down/up levels = len(ch_mult).
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 1, base: int = 64,
                 ch_mult=(1, 2, 2), cond_dim: int = 17, time_dim: int = 64,
                 emb_dim: int = 256, groups: int = 8):
        super().__init__()
        self.time_dim = time_dim
        self.emb_mlp = nn.Sequential(
            nn.Linear(time_dim + cond_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))

        chs = [base * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # encoder: ResBlock (save skip) then strided-conv downsample
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = base
        for c in chs:
            self.enc_blocks.append(ResBlock(prev, c, emb_dim, groups))
            self.downs.append(nn.Conv2d(c, c, 4, stride=2, padding=1))
            prev = c

        self.mid = ResBlock(prev, prev, emb_dim, groups)

        # decoder: transpose-conv upsample, concat skip, ResBlock
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 4, stride=2, padding=1))
            self.dec_blocks.append(ResBlock(c + c, c, emb_dim, groups))
            prev = c

        self.out_norm = nn.GroupNorm(min(groups, prev), prev)
        self.out_conv = nn.Conv2d(prev, out_ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, xt, t, cond_map, cosmo):
        # xt, cond_map: (B, 1, L, L); t: (B,); cosmo: (B, cond_dim)
        if xt.dim() == 3:
            xt = xt[:, None]
        if cond_map.dim() == 3:
            cond_map = cond_map[:, None]
        emb = self.emb_mlp(torch.cat(
            [sinusoidal_embedding(t, self.time_dim), cosmo], -1))
        h = self.in_conv(torch.cat([xt, cond_map], 1))
        skips = []
        for blk, dn in zip(self.enc_blocks, self.downs):
            h = blk(h, emb)
            skips.append(h)
            h = dn(h)
        h = self.mid(h, emb)
        for up, blk in zip(self.ups, self.dec_blocks):
            h = up(h)
            h = blk(torch.cat([h, skips.pop()], 1), emb)
        return self.out_conv(self.act(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Flow-matching loss and ODE sampler  (2D patch tensors)
# ---------------------------------------------------------------------------
def flow_matching_loss(net, x1, cond_map, cosmo):
    """Conditional rectified-flow loss. x1 = high-res patch (B,1,L,L); cond = DISCO
    patch (B,1,L,L) + cosmo (B,cond_dim)."""
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    v_pred = net(xt, t, cond_map, cosmo)
    return ((v_pred - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def sample_ode(net, cond_map, cosmo, steps: int = 50, x0=None):
    """Integrate dx/dt = v_theta from noise to a high-res patch realization (Euler)."""
    if cond_map.dim() == 3:
        cond_map = cond_map[:, None]
    x = torch.randn_like(cond_map) if x0 is None else x0
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((x.shape[0],), s * dt, device=x.device)
        x = x + net(x, t, cond_map, cosmo) * dt
    return x
