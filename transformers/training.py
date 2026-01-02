import torch
from typing import Dict


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute reconstruction + KL losses.
    """
    recon = torch.mean((outputs["reconstructed"] - targets) ** 2)

    mean = outputs["z_mean"]
    logvar = outputs["z_logvar"]

    kl = -0.5 * torch.mean(1 + logvar - mean**2 - logvar.exp())

    return {
        "loss": recon + 0.1 * kl,
        "recon": recon,
        "kl": kl,
    }
