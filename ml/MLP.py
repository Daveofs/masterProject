"""MLP model definition for flow matching."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, feature_dim: int = 1, cond_dim: int = 0, hidden: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, x, t, cond=None):
        """
        x: [Batch, N_alms, feature_dim]
        t: [Batch]
        cond: [Batch, cond_dim] or None
        """
        B, N_alms, _ = x.shape

        # Reshape t to [Batch, 1, 1] and expand across the N_alms dimension
        T = t.view(B, 1, 1).expand(B, N_alms, 1)

        if cond is None:
            inp = torch.cat([x, T], dim=-1)
        else:
            C = cond.view(B, 1, -1).expand(B, N_alms, cond.shape[-1])
            inp = torch.cat([x, T, C], dim=-1)

        return self.net(inp)

    def forward_chunked(self, x, t, cond=None, chunk_size=200_000):
        """
        Memory-efficient forward pass that chunks over the N_alms dimension.
        Mathematically identical to forward() but uses much less GPU memory.
        
        x: [Batch, N_alms, feature_dim]
        t: [Batch]
        cond: [Batch, cond_dim] or None
        chunk_size: number of alms to process at once
        """
        B, N_alms, _ = x.shape
        outputs = []

        for start in range(0, N_alms, chunk_size):
            end = min(start + chunk_size, N_alms)
            x_chunk = x[:, start:end, :]  # [B, chunk, feature_dim]
            out_chunk = self.forward(x_chunk, t, cond=cond)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=1)
