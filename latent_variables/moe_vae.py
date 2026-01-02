import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_dims, output_dim=None):
    layers = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers += [
            nn.Linear(dims[i], dims[i + 1]),
            nn.LayerNorm(dims[i + 1]),
            nn.ReLU()
        ]
    if output_dim is not None:
        layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class MoEVAE(nn.Module):
    """
    Mixture-of-Experts Variational Autoencoder.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = build_mlp(
            cfg.MOE_INPUT_DIM,
            cfg.ENCODER_HIDDEN_DIMS
        )

        h_dim = cfg.ENCODER_HIDDEN_DIMS[-1]

        self.router = nn.Sequential(
            nn.Linear(h_dim, cfg.NUM_EXPERTS),
            nn.LayerNorm(cfg.NUM_EXPERTS)
        )

        self.mu = nn.Linear(h_dim, cfg.NUM_EXPERTS * cfg.LATENT_DIM)
        self.logvar = nn.Linear(h_dim, cfg.NUM_EXPERTS * cfg.LATENT_DIM)

        self.decoder = build_mlp(
            cfg.LATENT_DIM,
            cfg.DECODER_HIDDEN_DIMS,
            cfg.MOE_INPUT_DIM
        )

    def forward(self, x):
        B = x.size(0)
        h = self.encoder(x)

        probs = F.softmax(self.router(h), dim=-1)
        topk_vals, topk_idx = torch.topk(probs, self.cfg.TOP_K, dim=-1)

        mu = self.mu(h).view(B, self.cfg.NUM_EXPERTS, self.cfg.LATENT_DIM)
        logvar = self.logvar(h).view(B, self.cfg.NUM_EXPERTS, self.cfg.LATENT_DIM)

        zs = []
        for k in range(self.cfg.TOP_K):
            idx = topk_idx[:, k]
            mu_k = mu[torch.arange(B), idx]
            lv_k = logvar[torch.arange(B), idx]
            z = mu_k + torch.randn_like(mu_k) * torch.exp(0.5 * lv_k)
            zs.append(z)

        z = (torch.stack(zs, dim=1) * topk_vals.unsqueeze(-1)).sum(dim=1)
        return self.decoder(z), mu, logvar, probs
