from abc import ABC, abstractmethod

from lightning_trainable.hparams import HParams

class NetworkHParams(HParams):
    """This HParams class is empty, but it serves as a type hint for all other network hparams."""

class BaseNetwork(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def inverse(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def dims_in(self):
        pass

    @property
    @abstractmethod
    def dims_out(self):
        pass
