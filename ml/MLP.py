"""FiLM-Modulated Conv1D model definition for flow matching.

Optimized with a global residual connection and standard initialization 
to break loss flatlines and actively map physical variations.
"""

import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for sinusoidal embedding.")
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]

        half_dim = self.embed_dim // 2
        device = t.device
        dtype = t.dtype

        freq = torch.exp(
            torch.linspace(
                0, math.log(10000.0), half_dim, device=device, dtype=dtype
            )
        )
        angles = t * freq[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class FiLMResidualConv1dBlock(nn.Module):
    def __init__(self, hidden: int, emb_dim: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        
        self.norm1 = nn.GroupNorm(1, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        self.film = nn.Linear(emb_dim, hidden * 2)
        
        self.norm2 = nn.GroupNorm(1, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding)

        # Standard initialization for internal layers
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.act(h)
        h = self.dropout(h)
        
        film_params = self.film(emb).unsqueeze(-1)
        scale, shift = torch.chunk(film_params, 2, dim=1)
        
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = self.conv2(h)
        return x + h


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        cond_dim: int = 0,
        hidden: int = 64,  
        num_blocks: int = 4,
        dropout: float = 0.0,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.cond_dim = cond_dim
        self.hidden = hidden

        padding = kernel_size // 2
        self.in_proj = nn.Conv1d(1, hidden, kernel_size=kernel_size, padding=padding)

        time_embed_dim = hidden
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        emb_dim = hidden
        self.context_mlp = nn.Sequential(
            nn.Linear(time_embed_dim + cond_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        self.blocks = nn.ModuleList(
            [FiLMResidualConv1dBlock(hidden, emb_dim, kernel_size, dropout) for _ in range(num_blocks)]
        )

        self.out_norm = nn.GroupNorm(1, hidden)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv1d(hidden, 1, kernel_size=1)

        # Actively initialize weights to spark immediate gradient flow
        nn.init.xavier_normal_(self.in_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)

    def forward(self, x, t, cond=None):
        # Save structural identity path for physical global residual connection
        x_residual = x  

        h = x.unsqueeze(1)  # [B, 1, D]
        h = self.in_proj(h)  # [B, hidden, D]

        if t.dim() == 1:
            t = t[:, None]
        t_emb = self.time_embed(t)
        
        if cond is not None:
            if cond.dim() == 1:
                cond = cond[:, None]
            context = torch.cat([t_emb, cond], dim=-1)
        else:
            context = t_emb
            
        emb = self.context_mlp(context)

        for block in self.blocks:
            h = block(h, emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_proj(h)  # [B, 1, D]
        h = h.squeeze(1)      # [B, D]

        # Return global residual mapping: Input identity + predicted adjustments
        return x_residual + h