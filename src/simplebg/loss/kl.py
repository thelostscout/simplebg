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
    return - (target_energy(x_generated) + log_det_j_samples)
