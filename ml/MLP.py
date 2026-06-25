"""Spherical Harmonic Flow-Matching model — memory-efficient for lmax=3000.

Dense Linear(9M, hidden) requires 36 GB at hidden=1024. This replaces it
with a per-ell block projection that exploits the (ell, m) structure of alms.

Memory at lmax=3000, d_ell=32, 6 blocks:  ~200 MB total  (vs 36 GB)
"""

import math
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        half = self.embed_dim // 2
        freq = torch.exp(
            torch.linspace(0, math.log(10000.0), half, device=t.device, dtype=t.dtype)
        )
        angles = t * freq.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class EllCouplingBlock(nn.Module):
    """
    Depthwise-separable 1-D conv over the ell axis.
    Couples adjacent scales; kernel=5 covers ~±2 ell steps.
    Parameters: d_ell*kernel (depthwise) + d_ell^2 (pointwise) — tiny.
    """

    def __init__(self, d_ell: int, emb_dim: int, kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.norm = nn.LayerNorm(d_ell)
        self.dw   = nn.Conv1d(d_ell, d_ell, kernel, padding=pad, groups=d_ell)
        self.pw   = nn.Conv1d(d_ell, d_ell, 1)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(emb_dim, d_ell * 2)
        nn.init.zeros_(self.pw.weight);   nn.init.zeros_(self.pw.bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # h: (B, L, d_ell),  emb: (B, emb_dim)
        r = h
        h = self.norm(h).transpose(1, 2)        # (B, d_ell, L)
        h = self.act(self.dw(h))
        h = self.drop(self.pw(h)).transpose(1, 2)  # (B, L, d_ell)
        s, b = self.film(emb).chunk(2, dim=-1)
        h = h * (1 + s.unsqueeze(1)) + b.unsqueeze(1)
        return r + h


class EllPointwiseBlock(nn.Module):
    """
    Same small MLP applied independently to each ell (= Conv1d kernel=1).
    expand=2 gives a hidden size of 2*d_ell inside the block.
    """

    def __init__(self, d_ell: int, emb_dim: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        mid = d_ell * expand
        self.norm  = nn.LayerNorm(d_ell)
        self.lin1  = nn.Linear(d_ell, mid)
        self.act   = nn.SiLU()
        self.drop  = nn.Dropout(dropout)
        self.lin2  = nn.Linear(mid, d_ell)
        self.film  = nn.Linear(emb_dim, mid * 2)
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin2.weight); nn.init.zeros_(self.lin2.bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        r = h
        h = self.norm(h)
        h = self.act(self.lin1(h))
        s, b = self.film(emb).chunk(2, dim=-1)
        h = h * (1 + s.unsqueeze(1)) + b.unsqueeze(1)
        h = self.drop(h)
        h = self.lin2(h)
        return r + h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Drop-in replacement for the original MLP.
    Same constructor signature (dim_in, cond_dim, hidden) + new lmax kwarg.
    'hidden' is reused as d_ell (per-ell embedding width); use 16–64.

    Required change in train_flow_matching.py:
        model = MLP(dim_in=dim_in, cond_dim=cond_dim,
                    hidden=args.hidden, lmax=args.lmax)   # add lmax=

    apply_flow_correction.py already reads lmax from metadata — no change needed.
    """

    def __init__(
        self,
        dim_in: int,
        cond_dim: int = 0,
        hidden: int = 32,       # d_ell; 32 is a good default for lmax=3000
        lmax: int = 3000,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_in   = dim_in
        self.cond_dim = cond_dim
        self.d_ell    = hidden
        self.lmax     = lmax
        N_alm = (lmax + 1) * (lmax + 2) // 2
        self.N_alm = N_alm
        self.M     = lmax + 1           # max m-modes per ell row

        if dim_in != 2 * N_alm:
            raise ValueError(
                f"dim_in={dim_in} but 2*N_alm(lmax={lmax})={2*N_alm}. "
                "Pass the correct lmax to MLP()."
            )

        # ---- index buffers (not parameters) ----
        re_idx, im_idx, ell_pos, mode_pos, _, _ = _build_ell_index_buffers(lmax)
        self.register_buffer("re_idx",   re_idx)
        self.register_buffer("im_idx",   im_idx)
        self.register_buffer("ell_pos",  ell_pos)
        self.register_buffer("mode_pos", mode_pos)

        # ---- time + context embedding ----
        t_dim   = max(64, hidden * 8)
        emb_dim = max(64, hidden * 8)
        self.time_embed  = SinusoidalTimeEmbedding(t_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(t_dim + cond_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        # ---- encoder / decoder ----
        # Shared linear over the padded 2*(lmax+1) per-ell block
        self.encoder = nn.Linear(2 * self.M, hidden)
        self.decoder = nn.Linear(hidden, 2 * self.M)
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        # ---- depth-interleaved blocks ----
        self.coupling   = nn.ModuleList([
            EllCouplingBlock(hidden, emb_dim, kernel=5, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.pointwise  = nn.ModuleList([
            EllPointwiseBlock(hidden, emb_dim, expand=2, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden)

    # ------------------------------------------------------------------
    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2*N_alm) → (B, lmax+1, 2*(lmax+1))"""
        B = x.shape[0]
        out = torch.zeros(B, self.lmax + 1, 2 * self.M,
                          device=x.device, dtype=x.dtype)
        out[:, self.ell_pos, self.mode_pos]          = x[:, self.re_idx]
        out[:, self.ell_pos, self.M + self.mode_pos] = x[:, self.im_idx]
        return out

    def _scatter(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, lmax+1, 2*(lmax+1)) → (B, 2*N_alm)"""
        B = h.shape[0]
        out = torch.zeros(B, 2 * self.N_alm,
                          device=h.device, dtype=h.dtype)
        out[:, self.re_idx]   = h[:, self.ell_pos, self.mode_pos]
        out[:, self.im_idx]   = h[:, self.ell_pos, self.M + self.mode_pos]
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor = None) -> torch.Tensor:
        """
        x:    (B, 2*N_alm)  whitened flat alm vector
        t:    (B,)           flow time ∈ [0, 1]
        cond: (B, cond_dim)
        → (B, 2*N_alm)  predicted velocity
        """
        # 1. Embedding
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.time_embed(t)                         # (B, t_dim)
        ctx   = torch.cat([t_emb, cond], dim=-1) if cond is not None else t_emb
        emb   = self.context_mlp(ctx)                      # (B, emb_dim)

        # 2. Gather → encode
        x_ell = self._gather(x)                            # (B, L, 2*M)
        h     = self.encoder(x_ell)                        # (B, L, d_ell)

        # 3. Interleaved blocks
        for coup, pw in zip(self.coupling, self.pointwise):
            h = coup(h, emb)
            h = pw(h, emb)

        # 4. Decode → scatter
        h = self.out_norm(h)                               # (B, L, d_ell)
        h = self.decoder(h)                                # (B, L, 2*M)
        v = self._scatter(h)                               # (B, 2*N_alm)
        return v