import torch
import torch.distributions as D
from typing import Union
from collections import namedtuple

import fff
from fff.utils.types import Transform

LossOutput = namedtuple("LossOutput", ["loss", "nll", "mse", "z", "x1"])


def general_loss(
        x: torch.Tensor,
        encode: Transform,
        decode: Transform,
        latent_distribution: D.Distribution,
        beta: Union[float, torch.Tensor],
        training: bool,
        hutchinson_samples: int = 1,
) -> LossOutput:
    if training:
        # compute the nll surrogate
        surrogate = fff.loss.volume_change_surrogate(x, encode, decode, hutchinson_samples)
        # compute the mse
        # note, that currently (12.06.2024), fff actually computes the squared error without taking the mean
        mse = fff.loss.reconstruction_loss(x, surrogate.x1)
        log_prob = latent_distribution.log_prob(surrogate.z)
        nll = - fff.loss.sum_except_batch(log_prob) - surrogate.surrogate
        # the loss itself is meaningless, but the gradients are correct for training
        # all tensors are of shape (batch_size,)
        return LossOutput(beta * mse + nll, nll, mse, surrogate.z, surrogate.x1)
    else:
        # for validation, we need to calculate the exact loss because the training loss itself is meaningless. This is
        # costly but ok if only done during validation steps
        # compute the exact nll
        exact = fff.other_losses.exact_nll(x, encode, decode, latent_distribution)
        # compute the mse
        # note, that currently (12.06.2024), fff actually computes the squared error without taking the mean
        mse = fff.loss.reconstruction_loss(x, exact.x1)
        # for validation, we only want the exact nll term without the mse, but we can track the mse as an additional
        # metric.
        return LossOutput(exact.nll, exact.nll, mse, exact.z, exact.x1)

