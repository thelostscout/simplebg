import fff
from lightning_trainable.hparams import AttributeDict
from torch import nn

from . import subnets, transforms
from .core import BaseNetwork, NetworkOutput, NetworkHParams


class FreeFormFlowHParams(NetworkHParams):
    network_module = "fff"


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
            self.hparams = self.hparams_type(**hparams)
        else:
            self.hparams = hparams
        self._dims_in = dims_in

    @property
    def dims_in(self):
        return self._dims_in

    @property
    def dims_out(self):
        raise NotImplementedError

    @property
    def encode(self, x, **kwargs):
        raise NotImplementedError

    @property
    def decode(self, x, **kwargs):
        raise NotImplementedError

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


class SubNetFlowHParams(FreeFormFlowHParams):
    network_class = "SubNetFlow"
    subnet_hparams: subnets.SubnetHParams


class SubNetFlow(BaseFreeFormFlow):
    hparams: SubNetFlowHParams
    hparams_type = SubNetFlowHParams

    def __init__(
            self,
            dims_in: int,
            hparams: SubNetFlowHParams | dict,
    ):
        super().__init__(dims_in=dims_in, hparams=hparams)
        subnet_hparams = self.hparams.subnet_hparams.copy()
        subnet_class = subnet_hparams.pop("subnet_class")
        SubNetClass = getattr(subnets, subnet_class)
        self.encoder_net = SubNetClass(dims_in=self.dims_in, dims_out=self.dims_out, **subnet_hparams)
        self.decoder_net = SubNetClass(dims_in=self.dims_out, dims_out=self.dims_in, **subnet_hparams)

    @property
    def dims_out(self):
        return self.dims_in

    @property
    def encode(self, x, **kwargs):
        return self.encoder_net(x, **kwargs)

    @property
    def decode(self, x, **kwargs):
        return self.decoder_net(z, **kwargs)


class AngleFlow(SubNetFlow):
    def __init__(
            self,
            dims_in: int,
            hparams: SubNetFlowHParams | dict,
    ):
        super().__init__(dims_in=dims_in, hparams=hparams)
        self.rotate = subnets.angles.AngleShift(dims_in)

    @property
    def encode(self, x, **kwargs):
        x = self.rotate.encode(x)
        x = subnets.angles.angle_to_embedded_flat(x)
        x = self.encoder_net(x, **kwargs)
        x = subnets.angles.embedded_flat_to_angle(x)
        return x

    @property
    def decode(self, x, **kwargs):
        x = subnets.angles.angle_to_embedded_flat(x)
        x = self.decoder_net(x, **kwargs)
        x = subnets.angles.embedded_flat_to_angle(x)
        x = self.rotate.decode(x)
        return x
