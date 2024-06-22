import bgmol.systems
import numpy as np
import torch
from torch import nn

import fff
from lightning_trainable.hparams import AttributeDict

from .core import BaseNetwork, NetworkOutput, NetworkHParams
from . import subnets, transforms
from .. import loss


class FreeFormFlowHParams(NetworkHParams):
    network_module = "fff"
    bottleneck: int
    transform: str = "identity"
    transform_kwargs: dict = dict()


class BaseFreeFormFlow(BaseNetwork, nn.Module):
    hparams: FreeFormFlowHParams
    hparams_type = FreeFormFlowHParams
    exact_invertible = False

    def __init__(
            self,
            dims_in: int,
            hparams: FreeFormFlowHParams | dict,
            **transform_kwargs,
    ):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams = self.hparams_type(**hparams)
        self._dims_in = dims_in
        self._dims_out = self.hparams.bottleneck
        self.transform = transforms.constructor(self.hparams.transform, **transform_kwargs,
                                                **self.hparams.transform_kwargs)

    @property
    def dims_in(self):
        return self._dims_in

    @property
    def dims_out(self):
        return self._dims_out

    def forward(self, x, jac=False, **kwargs):
        byproducts = AttributeDict()
        x, log_det_j = self.transform.forward(x)
        if jac:
            # calculating the jacobian is costly
            if self.training:
                # for training, we use the surrogate because it is cheaper to compute.
                # The surrogate itself is meaningless, but we can train on the gradients.
                surrogate = fff.loss.volume_change_surrogate(x, self._encode, self._decode, **kwargs)
                z = surrogate.z
                log_det_j += surrogate.surrogate
                byproducts.x1 = surrogate.x1
                byproducts.regularizations = surrogate.regularizations
            else:
                # for validation, we need to calculate the exact loss because the training loss itself is meaningless
                # and because the surrogate requires gradients.
                # This is costly but ok if only done during validation steps
                exact = fff.other_losses.volume_change_exact(x, self._encode, self._decode, **kwargs)
                z = exact.z
                log_det_j += exact.exact
                byproducts.x1 = exact.x1
                byproducts.regularizations = exact.regularizations
        else:
            z = self._encode(x)
            log_det_j = None
        return NetworkOutput(output=z, log_det_j=log_det_j, byproducts=byproducts)

    def reverse(self, z, jac=False, **kwargs):
        byproducts = AttributeDict()
        if jac:
            if self.training:
                surrogate = fff.loss.volume_change_surrogate(z, self._decode, self._encode, **kwargs)
                x = surrogate.z
                log_det_j = surrogate.surrogate
                byproducts.z1 = surrogate.x1
                byproducts.regularizations = surrogate.regularizations
            else:
                exact = fff.other_losses.volume_change_exact(z, self._decode, self._encode, **kwargs)
                x = exact.z
                log_det_j = exact.exact
                byproducts.z1 = exact.x1
                byproducts.regularizations = exact.regularizations
        else:
            x = self._decode(z)
            log_det_j = None
        x, log_det_j_transform = self.transform.reverse(x)
        if log_det_j is not None:
            log_det_j += log_det_j_transform
        return NetworkOutput(output=x, log_det_j=log_det_j, byproducts=byproducts)


class ConstWidthHParams(subnets.ConstWidthHParams):
    dropout: float = 0.
    residual: bool = True


class ResNetHParams(FreeFormFlowHParams):
    network_class = "ResNet"
    net_hparams: ConstWidthHParams | dict


class ResNet(BaseFreeFormFlow):
    hparams_type = ResNetHParams
    hparams: ResNetHParams

    def __init__(
            self,
            dims_in: int,
            hparams: ResNetHParams | dict,
            **transform_kwargs,
    ):
        super().__init__(dims_in=dims_in, hparams=hparams, **transform_kwargs)
        self._encode = subnets.ConstWidth(dims_in=dims_in, dims_out=self.dims_out, **self.hparams.net_hparams)
        self._decode = subnets.ConstWidth(dims_in=self.hparams.bottleneck, dims_out=dims_in, **self.hparams.net_hparams)


class ResNetICHParams(ResNetHParams):
    network_class = "ResNetIC"
    normalize_angles: bool
