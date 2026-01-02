import torch
import torch.nn.functional as F


def moe_vae_loss(x, recon, mu, logvar, router_probs, cfg):
    """
    Full MoE-VAE loss.
    """
    recon_loss = F.mse_loss(recon, x)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    entropy = -torch.mean(
        torch.sum(router_probs * torch.log(router_probs + cfg.EPS), dim=1)
    )

    load = router_probs.mean(dim=0)
    balance = torch.sum((load - 1.0 / cfg.NUM_EXPERTS) ** 2)

    total = (
        recon_loss
        + cfg.BETA_KL * kl
        + cfg.ROUTER_ENTROPY_WEIGHT * entropy
        + cfg.LOAD_BALANCE_WEIGHT * balance
    )

    return total, recon_loss, kl, entropy, balance
