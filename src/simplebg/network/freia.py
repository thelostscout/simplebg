from functools import partial
from typing import Iterable

import FrEIA
from FrEIA.framework import SequenceINN
from torch import Tensor

from . import core
from . import subnets


def subnet_constructor(dims_in, dims_out, subnet_name, **kwargs):
    SubnetClass = getattr(subnets, subnet_name)
    return SubnetClass(dims_in, dims_out, **kwargs)


class FrEIABaseHParams(core.NetworkHParams):
    network_module = "freia"


class FixedBlocksHParams(FrEIABaseHParams):
    network_class = "FixedBlocks"
    subnet_name: str
    subnet_hparams: subnets.SubnetHParams
    coupling_blocks: int
    coupling_block_name: str


class RNVPConstWidthHParams(FixedBlocksHParams):
    subnet_name = "ConstWidth"
    coupling_block_name = "AllInOneBlock"
    subnet_hparams: subnets.ConstWidthHParams


class RNVPExponentialHParams(FixedBlocksHParams):
    subnet_name = "Exponential"
    coupling_block_name = "AllInOneBlock"
    subnet_hparams: subnets.ExponentialHParams


class FrEIABase(core.BaseNetwork, SequenceINN):
    hparams_type = FrEIABaseHParams
    hparams: FrEIABaseHParams
    exact_invertible = True

    def __init__(
            self,
            dims_in: int,
            hparams: core.NetworkHParams | dict,
            **kwargs,
    ):
        if isinstance(hparams, dict):
            hparams = self.hparams_type(**hparams)
        super().__init__(dims_in)
        self.network_constructor(hparams=hparams)

    def forward(self, x: Tensor, c: Iterable[Tensor] = None, jac=True, **kwargs) -> core.NetworkOutput:
        return core.NetworkOutput(*SequenceINN.forward(self, x_or_z=x, c=c, rev=False, jac=jac), dict())

    def reverse(self, z: Tensor, c: Iterable[Tensor] = None, jac=True, **kwargs) -> core.NetworkOutput:
        return core.NetworkOutput(*SequenceINN.forward(self, x_or_z=z, c=c, rev=True, jac=jac), dict())

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
