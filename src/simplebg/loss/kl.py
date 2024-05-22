# This implements the Kullback-Leibler divergence, also known as negative log likelihood (nll).

from typing import Any
import torch
from lightning_trainable.hparams import AttributeDict
from .. import network
from .. import latent


def forward_kl(metrics: AttributeDict, **kwargs):
    latent_distribution = kwargs["latent_distribution"]
    return - (latent_distribution.log_prob(metrics.z) + metrics.log_det_j_forward)


def reverse_kl(metrics: AttributeDict, **kwargs):
    target_energy = kwargs["target_energy"]
    return - (target_energy(metrics.x_generated) + metrics.log_det_j_samples)
