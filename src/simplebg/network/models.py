import torch.nn as nn
from abc import ABC, abstractmethod
import torch.distributions as D
class BaseModel(nn.module, ABC):

    @property
    def q(self) -> D.Distribution:
        """
            The latent distribution of the flow. Anything that behaves like a D.Distribution in the sense that it has
            a q.sample(n_samples) method and a q.log_prob(tensor) method will be accepted. To turn your custom
            distribution into a D.Distribution, see the .distributions module.
        """
        return self._q

    def sample(self, n_samples: int):
        self.q.sample(n_samples)

    @abstractmethod
    def sample(self, n_samples: int):
        pass

