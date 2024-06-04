from abc import ABC, abstractmethod

import torch
import torch.distributions as D
from torch import Tensor

from lightning_trainable.hparams import HParams


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass


class DistributionHParams(HParams):
    name: str
    kwargs: dict


class Normal(D.MultivariateNormal):
    def __init__(
            self,
            dims: int,
            sigma: float | Tensor = 1.0,
    ):
        if isinstance(sigma, float):
            sigma = torch.ones(dims) * sigma
        else:
            if sigma.shape != (dims,):
                raise ValueError(f"sigma must have shape ({dims},), but has shape {sigma.shape}.")
        super().__init__(loc=torch.zeros(dims), covariance_matrix=torch.diag(sigma ** 2))
