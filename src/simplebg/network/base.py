from abc import ABC, abstractmethod

from lightning_trainable.hparams import HParams

from collections import namedtuple

NetworkOutput = namedtuple("NetworkOutput", ["output", "log_det_j"])


class NetworkHParams(HParams):
    """This HParams class is empty, but it serves as a type hint for all other network hparams."""


class BaseNetwork(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> NetworkOutput:
        pass

    @abstractmethod
    def reverse(self, *args, **kwargs) -> NetworkOutput:
        pass

    @property
    @abstractmethod
    def dims_in(self):
        pass

    @property
    @abstractmethod
    def dims_out(self):
        pass
