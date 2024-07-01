import fff
from lightning_trainable.hparams import AttributeDict
from torch import nn

from . import subnets, transforms
from .core import BaseNetwork, NetworkOutput, NetworkHParams


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
        else:
            self.hparams = hparams
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


class SubNetFreeFormFlowHParams(FreeFormFlowHParams):
    network_class = "SubNetFreeFormFlow"
    subnet_hparams: subnets.SubnetHParams


class SubNetFreeFormFlow(BaseFreeFormFlow):
    def __init__(
            self,
            dims_in: int,
            hparams: SubNetFreeFormFlowHParams | dict,
            **transform_kwargs,
    ):
        super().__init__(dims_in=dims_in, hparams=hparams, **transform_kwargs)
        subnet_hparams = self.hparams.subnet_hparams
        subnet_class = subnet_hparams.pop("subnet_class")
        SubNetClass = getattr(subnets, subnet_class)
        self._encode = SubNetClass(dims_in=self.dims_in, dims_out=self.dims_out, **subnet_hparams)
        self._decode = SubNetClass(dims_in=self.dims_out, dims_out=self.dims_in, **subnet_hparams)
