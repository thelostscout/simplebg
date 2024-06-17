import torch
from torch import nn

import fff
from lightning_trainable.hparams import AttributeDict

from .core import BaseNetwork, NetworkOutput, NetworkHParams
from . import subnets
from .. import loss


class FreeFormFlowHParams(NetworkHParams):
    network_module = "fff"
    bottleneck: int


class BaseFreeFormFlow(BaseNetwork, nn.Module):
    hparams: FreeFormFlowHParams
    hparams_type = FreeFormFlowHParams
    exact_invertible = False

    def __init__(
            self,
            dims_in: int,
            hparams: FreeFormFlowHParams | dict,
    ):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = self.hparams_type(**hparams)
        self._dims_in = dims_in
        self._dims_out = hparams.bottleneck

    @property
    def dims_in(self):
        return self._dims_in

    @property
    def dims_out(self):
        return self._dims_out

    def forward(self, x, jac=False, **kwargs):
        byproducts = AttributeDict()
        if jac:
            # calculating the jacobian is costly
            if self.training:
                # for training, we use the surrogate because it is cheaper to compute.
                # The surrogate itself is meaningless, but we can train on the gradients.
                surrogate = fff.loss.volume_change_surrogate(x, self.encode, self.decode, **kwargs)
                z = surrogate.z
                log_det_j = surrogate.surrogate
                byproducts.x1 = surrogate.x1
                byproducts.regularizations = surrogate.regularizations
            else:
                # for validation, we need to calculate the exact loss because the training loss itself is meaningless
                # and because the surrogate requires gradients.
                # This is costly but ok if only done during validation steps
                exact = fff.other_losses.volume_change_exact(x, self.encode, self.decode, **kwargs)
                z = exact.z
                log_det_j = exact.exact
                byproducts.x1 = exact.x1
                byproducts.regularizations = exact.regularizations
        else:
            z = self.encode(x)
            log_det_j = None
        return NetworkOutput(output=z, log_det_j=log_det_j, byproducts=byproducts)

    def reverse(self, z, jac=False, **kwargs):
        byproducts = AttributeDict()
        if jac:
            if self.training:
                surrogate = fff.loss.volume_change_surrogate(z, self.decode, self.encode, **kwargs)
                x = surrogate.z
                log_det_j = surrogate.surrogate
                byproducts.z1 = surrogate.x1
                byproducts.regularizations = surrogate.regularizations
            else:
                exact = fff.other_losses.volume_change_exact(z, self.decode, self.encode, **kwargs)
                x = exact.z
                log_det_j = exact.exact
                byproducts.z1 = exact.x1
                byproducts.regularizations = exact.regularizations
        else:
            x = self.decode(z)
            log_det_j = None
        return NetworkOutput(output=x, log_det_j=log_det_j, byproducts=byproducts)


class ConstWidthHParams(subnets.ConstWidthHParams):
    dropout: float = 0.
    residual: bool = True


class ResNetHParams(FreeFormFlowHParams):
    network_class = "ResNet"
    net_hparams: ConstWidthHParams | dict


class ResNet(BaseFreeFormFlow):
    hparams_type = ResNetHParams

    def __init__(
            self,
            dims_in: int,
            hparams: ResNetHParams | dict,
    ):
        super().__init__(dims_in=dims_in, hparams=hparams)
        self.encode = subnets.ConstWidth(dims_in=dims_in, dims_out=self.dims_out, **hparams.net_hparams)
        self.decode = subnets.ConstWidth(dims_in=hparams.bottleneck, dims_out=dims_in, **hparams.net_hparams)
