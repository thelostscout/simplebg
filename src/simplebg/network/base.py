import torch
from torch import Tensor
from torch import nn
from abc import ABC, abstractmethod
import torch.distributions as D
from typing import Tuple


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass


class BaseNetwork(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def inverse(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def input_dims(self):
        pass

    @property
    @abstractmethod
    def output_dims(self):
        pass
