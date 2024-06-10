from abc import ABC, abstractmethod
from collections import namedtuple

from torch import Tensor

from lightning_trainable.hparams import HParams


NetworkOutput = namedtuple("NetworkOutput", ["output", "log_det_j"])


class NetworkHParams(HParams):
    network_module: str
    network_class: str


class BaseNetwork(ABC):
    @abstractmethod
    def forward(self, x: Tensor, *args, jac=True, **kwargs) -> NetworkOutput:
        pass

    @abstractmethod
    def reverse(self, z: Tensor, *args, jac=True, **kwargs) -> NetworkOutput:
        pass

    @property
    @abstractmethod
    def dims_in(self):
        pass

    @property
    @abstractmethod
    def dims_out(self):
        pass