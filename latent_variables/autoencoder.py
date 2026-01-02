import torch
import torch.nn as nn
from typing import List


class DeepAutoEncoder(nn.Module):
    """
    Deep symmetric autoencoder.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_multipliers: List[int],
        dropout: float
    ):
        super().__init__()

        enc = []
        prev = input_dim
        for m in hidden_multipliers:
            h = latent_dim * m
            enc += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h

        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for m in reversed(hidden_multipliers):
            h = latent_dim * m
            dec += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h

        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
