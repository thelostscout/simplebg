# This implements the Kullback-Leibler divergence, also known as negative log likelihood (nll).
import torch
from lightning_trainable.hparams import AttributeDict
from .. import latent


def forward_kl(
        z: torch.Tensor,
        log_det_j_forward: torch.Tensor,
        latent_distribution: latent.BaseDistribution,
):
    return - (latent_distribution.log_prob(z) + log_det_j_forward)


def reverse_kl(
        x_generated: torch.Tensor,
        log_det_j_samples: torch.Tensor,
        target_energy,
):
    energies = target_energy(x_generated).squeeze(-1)
    # Clamp energies to avoid numerical issues
    capped_energies = torch.clamp(torch.nan_to_num(energies, nan=0.), max=1e6)
    capped_determinants = torch.clamp(torch.nan_to_num(log_det_j_samples, nan=0.), max=1e6)
    return - (capped_energies + log_det_j_samples)
