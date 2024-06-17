from abc import ABC, abstractmethod
from collections import namedtuple

from torch import Tensor

from lightning_trainable.hparams import HParams

NetworkOutput = namedtuple("NetworkOutput", ["output", "log_det_j", "byproducts"])


class NetworkHParams(HParams):
    network_module: str
    network_class: str


class BaseNetwork(ABC):
    exact_invertible: bool

    @abstractmethod
    def forward(self, x: Tensor, *args, jac=True, **kwargs) -> NetworkOutput:
        raise NotImplementedError

    @abstractmethod
    def reverse(self, z: Tensor, *args, jac=True, **kwargs) -> NetworkOutput:
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_in(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dims_out(self):
        raise NotImplementedError
