import torch
from torch import nn

from .core import BaseNetwork, NetworkOutput, NetworkHParams
from . import subnets
from .. import loss


class ConstWidthHParams(subnets.ConstWidthHParams):
    dropout: float = .05
    residual: bool = True


class ResNetHParams(NetworkHParams):
    network_module = "resnet"
    network_class = "ResNetSimple"
    bottleneck: int
    net_hparams: ConstWidthHParams | dict


class ResNetSimple(BaseNetwork, nn.Module):
    def __init__(
            self,
            dims_in: int,
            hparams: ResNetHParams | dict,
    ):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = ResNetHParams(**hparams)
        self._dims_in = dims_in
        self._dims_out = hparams.bottleneck
        self.encode = subnets.ConstWidth(dims_in=dims_in, dims_out=self.dims_out, **hparams.net_hparams)
        self.decode = subnets.ConstWidth(dims_in=hparams.bottleneck, dims_out=dims_in, **hparams.net_hparams)

    @property
    def dims_in(self):
        return self._dims_in

    @property
    def dims_out(self):
        return self._dims_out

    def forward(self, x, jac=False, **kwargs):
        z = self.encode(x)
        if jac:
            x1, jac = loss.fff.compute_jacobian(
                z,
                lambda y: self.decode(y)[0],
            )
            log_det_j = torch.slogdet(jac).logabsdet
        else:
            log_det_j = None

        return NetworkOutput(output=z, log_det_j=log_det_j)

    def reverse(self, z, jac=False, **kwargs):
        if jac:
            x1, jac = loss.fff.compute_jacobian(
                z,
                lambda y: self.decode(y)[0],
            )
            log_det_j = torch.slogdet(jac).logabsdet
        else:
            x1 = self.decode(z)
            log_det_j = None
        return NetworkOutput(output=x1, log_det_j=log_det_j)
