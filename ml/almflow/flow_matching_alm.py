"""Spherical-Harmonic (alm) Flow-Matching model — memory-efficient for lmax=3000.

Restored + extended from the original MLP.py. The model is a per-ell block MLP:
a dense Linear(2*N_alm, hidden) would need ~36 GB at lmax=3000; instead alms are
gathered into a padded (L, 2*M) block tensor and a shared MLP is applied across
ell rows (~200 MB total).

Flow matching in alm space: x0 = low alms, x1 = high alms (flat [Re||Im] vectors),
predict velocity v_theta(xt, t, cond) ~ x1 - x0, integrate the ODE from the low
alms to corrected alms, then alm2map to a corrected shell.

PER-ELL WHITENING (the fix for "small AND large scales don't match")
--------------------------------------------------------------------
Raw alms span a huge dynamic range (monopole Cl~7e3 vs ~1e-2 at ell>0), so an MSE
loss on raw alms is dominated by the monopole/low-ell and barely learns high-ell
corrections. We whiten each alm by a per-ell scale s(ell)=sqrt(<Cl(ell)>) so the
network sees O(1) inputs at every angular scale and the loss weights all ell
comparably. Training/inference happen in whitened space; unwhiten before alm2map.
"""

import math
import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# alm layout helpers
# ---------------------------------------------------------------------------

def _build_ell_index_buffers(lmax: int):
    """Index tensors for vectorized gather/scatter between the flat alm vector
    (healpy m-major, [Re || Im]) and a padded (L, 2*M) block tensor."""
    N_alm = (lmax + 1) * (lmax + 2) // 2
    M = lmax + 1
    re_idx, im_idx, ell_pos, mode_pos = [], [], [], []
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            idx = m * (2 * lmax + 1 - m) // 2 + ell
            re_idx.append(idx)
            im_idx.append(N_alm + idx)
            ell_pos.append(ell)
            mode_pos.append(m)
    return (torch.tensor(re_idx, dtype=torch.long),
            torch.tensor(im_idx, dtype=torch.long),
            torch.tensor(ell_pos, dtype=torch.long),
            torch.tensor(mode_pos, dtype=torch.long), N_alm, M)


def ell_of_flat_index(lmax: int) -> np.ndarray:
    """ell value for each of the N_alm complex modes (healpy m-major ordering)."""
    ell = np.empty((lmax + 1) * (lmax + 2) // 2, dtype=np.int64)
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            ell[m * (2 * lmax + 1 - m) // 2 + l] = l
    return ell


def whiten_scale_vector(cl_ref: np.ndarray, lmax: int, floor: float = 1e-12) -> np.ndarray:
    """Per-component scale for the flat [Re||Im] alm vector from a reference Cl(ell).

    Returns a (2*N_alm,) float32 vector; dividing alms by it whitens per ell.
    """
    s_ell = np.sqrt(np.maximum(cl_ref[:lmax + 1], floor)).astype(np.float32)
    ell = ell_of_flat_index(lmax)
    s = s_ell[ell]                       # (N_alm,)
    return np.concatenate([s, s]).astype(np.float32)   # Re and Im share the scale


# ---------------------------------------------------------------------------
# Model (faithful restore)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim_in, cond_dim=0, hidden=64, lmax=3000):
        super().__init__()
        self.lmax = lmax
        M = lmax + 1
        N_alm = (lmax + 1) * (lmax + 2) // 2
        self.N_alm = N_alm
        self.M = M

        re_idx, im_idx, ell_pos, mode_pos, _, _ = _build_ell_index_buffers(lmax)
        self.register_buffer("re_idx", re_idx)
        self.register_buffer("im_idx", im_idx)
        self.register_buffer("ell_pos", ell_pos)
        self.register_buffer("mode_pos", mode_pos)
        self.register_buffer("ell_index",
                             torch.arange(lmax + 1, dtype=torch.float32) / max(lmax, 1))

        # Conditional generative: input per ell = [xt_ell, x_low_ell, t, ell, cond].
        # The low alms are fed as conditioning so the model reproduces the (near-
        # deterministic) large scales AND generates the stochastic small scales.
        per_ell_in = 2 * M + 2 * M + 2 + cond_dim  # xt, x_low, t, ell index, cond
        self.net = nn.Sequential(
            nn.Linear(per_ell_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * M),
        )

    def _gather(self, x):
        B = x.shape[0]
        out = torch.zeros(B, self.lmax + 1, 2 * self.M, device=x.device, dtype=x.dtype)
        out[:, self.ell_pos, self.mode_pos] = x[:, self.re_idx]
        out[:, self.ell_pos, self.M + self.mode_pos] = x[:, self.im_idx]
        return out

    def _scatter(self, h):
        B = h.shape[0]
        out = torch.zeros(B, 2 * self.N_alm, device=h.device, dtype=h.dtype)
        out[:, self.re_idx] = h[:, self.ell_pos, self.mode_pos]
        out[:, self.im_idx] = h[:, self.ell_pos, self.M + self.mode_pos]
        return out

    def forward(self, x, t, cond=None, x_low=None):
        B = x.shape[0]
        L = self.lmax + 1
        x_ell = self._gather(x)                                  # noisy state xt
        xl_ell = self._gather(x_low) if x_low is not None else torch.zeros_like(x_ell)
        t_exp = t.view(B, 1, 1).expand(B, L, 1)
        ell_exp = self.ell_index.view(1, L, 1).expand(B, L, 1)
        parts = [x_ell, xl_ell, t_exp, ell_exp]
        if cond is not None:
            parts.append(cond.unsqueeze(1).expand(B, L, -1))
        inp = torch.cat(parts, dim=-1)
        h = self.net(inp)
        return self._scatter(h)


# ---------------------------------------------------------------------------
# Flow matching: loss + ODE sampler (operate in WHITENED alm space)
# ---------------------------------------------------------------------------

def flow_matching_loss(model, x_low, x_high, cond, sigma=0.0):
    """Conditional generative rectified-flow loss on the RESIDUAL (WHITENED alms).

    Target is r = x_high - x_low; x0 ~ N(0, I); conditioned on x_low. The model
    transports noise -> residual | low. Large scales (near-deterministic residual)
    are learned as the conditional mean; small scales (stochastic residual) are
    *generated*. corrected = low + sampled residual, so large scales are preserved.
    """
    B = x_high.shape[0]
    r = x_high - x_low
    x0 = torch.randn_like(r)
    t = torch.rand(B, device=r.device)
    xt = (1 - t)[:, None] * x0 + t[:, None] * r
    v_pred = model(xt, t, cond, x_low=x_low)
    return ((v_pred - (r - x0)) ** 2).mean()


@torch.no_grad()
def integrate(model, x_low, cond, steps=25, x0=None):
    """Sample corrected (whitened) alms = low + generated residual (noise->r|low)."""
    x = torch.randn_like(x_low) if x0 is None else x0
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((x.shape[0],), s * dt, device=x.device)
        x = x + model(x, t, cond, x_low=x_low) * dt
    return x_low + x                      # preserve large scales; add correction
