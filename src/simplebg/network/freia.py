from functools import partial
from typing import Iterable

import FrEIA
from FrEIA.framework import SequenceINN
from torch import Tensor

from . import base
from . import subnets


def subnet_constructor(dims_in, dims_out, subnet_name, **kwargs):
    SubnetClass = getattr(subnets, subnet_name)
    return SubnetClass(dims_in, dims_out, **kwargs)


class FixedBlocksHParams(base.NetworkHParams):
    subnet_name: str
    subnet_hparams: subnets.SubnetHParams
    coupling_blocks: int
    coupling_block_name: str


class RNVPLinearHParams(FixedBlocksHParams):
    subnet_name = "LinearSubnet"
    coupling_block_name = "AllInOneBlock"
    subnet_hparams: subnets.LinearSubnetHParams


class RNVPExponentialHParams(FixedBlocksHParams):
    subnet_name = "ExponentialSubnet"
    coupling_block_name = "AllInOneBlock"
    subnet_hparams: subnets.ExponentialSubnetHParams


class FrEIABase(base.BaseNetwork, SequenceINN):
    hparams_type = base.NetworkHParams
    hparams: base.NetworkHParams
    def __init__(
            self,
            dims_in: int,
            hparams: base.NetworkHParams | dict,
    ):
        if isinstance(hparams, dict):
            hparams = self.hparams_type(**hparams)
        super().__init__(dims_in)
        self.network_constructor(hparams=hparams)

    def forward(self, x: Tensor, c: Iterable[Tensor] = None):
        return SequenceINN.forward(self, x_or_z=x, c=c, rev=False, jac=True)

    def inverse(self, z: Tensor, c: Iterable[Tensor] = None):
        return SequenceINN.forward(self, x_or_z=z, c=c, rev=True, jac=True)

    @property
    def dims_in(self):
        return self._dims_in

    @dims_in.setter
    def dims_in(self, value):
        self._dims_in = value

    @dims_in.deleter
    def dims_in(self):
        del self._dims_in

    @property
    def dims_out(self):
        return self.shapes[-1]

    def network_constructor(self, **kwargs):
        raise NotImplementedError


class FixedBlocks(FrEIABase):
    hparams_type = FixedBlocksHParams
    hparams: FixedBlocksHParams

    def network_constructor(self, hparams: FixedBlocksHParams):
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
