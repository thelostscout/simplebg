from abc import ABC, abstractmethod

import bgflow
import torch
import torch.distributions as D
from torch import Tensor

import bgflow
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


class PriorWrapper(BaseDistribution):
    def __init__(
            self,
            prior: bgflow.ProductDistribution,
    ):
        self._prior = prior

    def sample(self, n_samples: int) -> tuple[torch.Tensor, ...]:
        return self._prior.sample(n_samples)

    def log_prob(self, value: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._prior.energy(*value)