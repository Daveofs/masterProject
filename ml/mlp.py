"""MLP model definition for flow matching."""

import torch
from torch import nn


# 3-layer MLP (3 layers of neurons)
class SmallMLP(nn.Module):
    # hidden is the width (number of units) of the intermediate MLP layers -> it directly controls the model capacity, compute and memory.
    def __init__(self, dim_in: int, cond_dim: int = 0, hidden=512):
        super().__init__()
        # [xt | t | z]  →  Linear → ReLU → Linear → ReLU → Linear → output
        self.net = nn.Sequential(
            nn.Linear(dim_in + 1 + cond_dim, hidden),  # applies linear transformation y = xA^T + b to the input x with shape (batch_size, dim_in + 1 + cond_dim) to the output y with shape (batch_size, hidden)
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_in),  # output layer
        )

    def forward(self, x, t, cond=None):
        # x: [B, D], t: [B]  cond: [B, C]
        T = t.view(-1, 1)
        if cond is None:
            inp = torch.cat([x, T], dim=-1)
        else:
            inp = torch.cat([x, T, cond], dim=-1)
        return self.net(inp)
