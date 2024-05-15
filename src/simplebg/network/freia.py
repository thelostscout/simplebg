from functools import partial
from typing import Iterable

from torch import Tensor

import FrEIA
from FrEIA.framework import SequenceINN
from lightning_trainable.hparams import HParams

from . import base
from . import subnets


def subnet_constructor(dims_in, dims_out, subnet_name, **kwargs):
    SubnetClass = getattr(subnets, subnet_name)
    return SubnetClass(dims_in, dims_out, **kwargs)


class FrEIAHParams(HParams):
    subnet_name: str
    subnet_hparams: HParams
    coupling_blocks: int
    coupling_block_name: str


class RNVPLinearHParams(FrEIAHParams):
    subnet_name = "LinearSubnet"
    coupling_block_name = "AllInOneBlock"
    subnet_hparams: subnets.LinearSubnetHParams


class RNVPExponentialHParams(FrEIAHParams):
    subnet_name = "ExponentialSubnet"
    coupling_block_name = "AllInOneBlock"
    subnet_hparams: subnets.ExponentialSubnetHParams


class FrEIABase(base.BaseNetwork, SequenceINN):
    def __init__(
            self,
            dims: int,
            hparams: FrEIAHParams | dict,
    ):
        if isinstance(hparams, dict):
            hparams = FrEIAHParams(**hparams)
        super().__init__(dims)
        self.network_constructor(hparams=hparams)

    def forward(self, x: Tensor, c: Iterable[Tensor] = None):
        return SequenceINN.forward(self, x_or_z=x, c=c, rev=False, jac=True)

    def inverse(self, z: Tensor, c: Iterable[Tensor] = None):
        return SequenceINN.forward(self, x_or_z=z, c=c, rev=True, jac=True)

    @property
    def input_dims(self):
        return self.shapes[0]

    @property
    def output_dims(self):
        return self.shapes[-1]

    def network_constructor(self, **kwargs):
        raise NotImplementedError


class FixedBlock(FrEIABase):
    def network_constructor(self, hparams: FrEIAHParams):
        Block = getattr(FrEIA.modules, hparams.coupling_block_name)
        if not Block:
            raise ValueError(f"Block {hparams.coupling_block_name} not found in FrEIA.modules.")
        for i in range(hparams.coupling_blocks):
            self.append(
                module_class=Block,
                subnet_constructor=partial(
                    subnet_constructor,
                    subnet_name=hparams.subnet_name,
                    **hparams.subnet_hparams
                ),
            )
