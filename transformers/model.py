import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from transformers.config import HyperParams
from transformers.time_encoding import MultiScaleTimeEncoder


class ThreadDynamics(nn.Module):
    """
    Learnable decay and boost parameters per thread.
    """

    def __init__(self, K: int):
        super().__init__()
        self.log_decay = nn.Parameter(torch.randn(K) * 0.1 - 2.0)
        self.log_boost = nn.Parameter(torch.randn(K) * 0.1)

    def decay_rates(self) -> torch.Tensor:
        return torch.exp(self.log_decay)

    def boost_rates(self) -> torch.Tensor:
        return torch.exp(self.log_boost)


class TransformerThreadEncoder(nn.Module):
    """
    Maps search embeddings + time embeddings → latent thread assignments.
    """

    def __init__(self, hp: HyperParams):
        super().__init__()
        self.hp = hp

        self.input_proj = nn.Linear(hp.embed_dim, hp.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model,
            nhead=hp.nhead,
            dim_feedforward=hp.dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, hp.num_layers)

        self.thread_head = nn.Linear(hp.d_model, hp.K * 2)
        self.signal_head = nn.Sequential(
            nn.Linear(hp.d_model, hp.d_model // 2),
            nn.GELU(),
            nn.Linear(hp.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        search_embeds: torch.Tensor,
        time_embeds: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ):
        x = self.input_proj(search_embeds) + time_embeds
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        stats = self.thread_head(x)
        mean, logvar = stats.chunk(2, dim=-1)

        if self.training:
            eps = torch.randn_like(mean)
            z = mean + eps * torch.exp(0.5 * logvar)
        else:
            z = mean

        probs = F.softmax(z, dim=-1)
        signal = self.signal_head(x)

        return mean, logvar, probs, signal


class DynamicThreadModel(nn.Module):
    """
    Full Dynamic Thread Transformer model.
    """

    def __init__(self, hp: HyperParams):
        super().__init__()
        self.hp = hp

        self.time_encoder = MultiScaleTimeEncoder(hp.d_model, hp.time_features)
        self.encoder = TransformerThreadEncoder(hp)
        self.decoder = nn.Linear(hp.K, hp.embed_dim)
        self.dynamics = ThreadDynamics(hp.K)

    def forward(
        self,
        embeds: torch.Tensor,
        timestamps: torch.Tensor,
        deltas: torch.Tensor,
        first_timestamp: float,
        padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        time_embeds = self.time_encoder(timestamps, deltas, first_timestamp)
        mean, logvar, probs, signal = self.encoder(
            embeds, time_embeds, padding_mask
        )
        recon = self.decoder(probs)

        return {
            "reconstructed": recon,
            "z_mean": mean,
            "z_logvar": logvar,
            "z_probs": probs,
            "signal_scores": signal,
        }
