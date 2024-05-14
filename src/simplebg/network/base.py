import torch
from torch import Tensor
from torch import nn
from abc import ABC, abstractmethod
import torch.distributions as D
from typing import Tuple


class BaseModel(nn.Module, ABC):

    @property
    def q(self) -> D.Distribution:
        """
            The latent distribution of the flow. Anything that behaves like a D.Distribution in the sense that it has
            a q.sample(n_samples) method and a q.log_prob(tensor) method will be accepted. To turn your custom
            distribution into a D.Distribution, see the .distributions module.
        """
        return self._q

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        z = self.q.sample(sample_shape)
        x = self.nn.inverse(z)[0]
        return x

    def log_prob(self, x):
        z, log_det_jf = self.nn.forward(x)
        return log_det_jf + self.q.log_prob(z)

    @abstractmethod
    @property
    def nn(self):
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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

    @abstractmethod
    @property
    def dims_in(self):
        pass

    @abstractmethod
    @property
    def dims_out(self):
        pass
