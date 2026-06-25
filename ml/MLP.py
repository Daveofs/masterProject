"""Spherical Harmonic Flow-Matching model — memory-efficient for lmax=3000.

Dense Linear(9M, hidden) requires 36 GB at hidden=1024. This replaces it
with a per-ell block projection that exploits the (ell, m) structure of alms.

Memory at lmax=3000, d_ell=32, 6 blocks:  ~200 MB total  (vs 36 GB)
"""

import math
import torch
from torch import nn


def _build_ell_index_buffers(lmax: int):
    """
    Returns four 1-D LongTensors for vectorized gather/scatter.

    The flat alm vector has layout:
      x = [Re(a_00), Re(a_10), Re(a_11), ...,  <- first N_alm entries
           Im(a_00), Im(a_10), Im(a_11), ...]   <- second N_alm entries

    Healpy m-major ordering idx = m * (2*lmax + 1 - m) // 2 + ell

    For each (ell, m) pair we record:
      re_idx   : index into the Re half of x
      im_idx   : index into the Im half of x  (= re_idx + N_alm)
      ell_pos  : which row in the padded (L, 2*M) block tensor
      mode_pos : which column in [0..M) for Re, [M..2M) for Im
    """
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
    return (
        torch.tensor(re_idx,   dtype=torch.long),
        torch.tensor(im_idx,   dtype=torch.long),
        torch.tensor(ell_pos,  dtype=torch.long),
        torch.tensor(mode_pos, dtype=torch.long),
        N_alm,
        M,
    )
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

        # No sinusoidal embedding — just raw t scalar
        per_ell_in = 2 * M + 1 + cond_dim

        self.net = nn.Sequential(
            nn.Linear(per_ell_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * M),
        )

    def _gather(self, x):
        B = x.shape[0]
        out = torch.zeros(B, self.lmax + 1, 2 * self.M, device=x.device, dtype=x.dtype)
        out[:, self.ell_pos, self.mode_pos]          = x[:, self.re_idx]
        out[:, self.ell_pos, self.M + self.mode_pos] = x[:, self.im_idx]
        return out

    def _scatter(self, h):
        B = h.shape[0]
        out = torch.zeros(B, 2 * self.N_alm, device=h.device, dtype=h.dtype)
        out[:, self.re_idx] = h[:, self.ell_pos, self.mode_pos]
        out[:, self.im_idx] = h[:, self.ell_pos, self.M + self.mode_pos]
        return out

    def forward(self, x, t, cond=None):
        B = x.shape[0]
        L = self.lmax + 1

        x_ell = self._gather(x)                              # (B, L, 2*M)

        # Broadcast scalar t to every ell row
        t_exp = t.view(B, 1, 1).expand(B, L, 1)             # (B, L, 1)
        parts = [x_ell, t_exp]
        if cond is not None:
            parts.append(cond.unsqueeze(1).expand(B, L, -1))

        inp = torch.cat(parts, dim=-1)                       # (B, L, 2*M+1+C)
        h = self.net(inp)                                    # (B, L, 2*M)
        return self._scatter(h)
