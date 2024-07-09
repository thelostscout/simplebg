from torch import nn

from .buildingblocks import ConstWidth, NormalizerFreeConstWidth, ScaledWSLinear
from .core import SubnetHParams


class ResNetHParams(SubnetHParams):
    subnet_class = "ResNet"
    depth_scheme: list[int]
    width_scheme: list[int]
    batch_norm: bool = True
    activation: nn.Module = nn.ReLU()


class ResNet(nn.Sequential):
    def __init__(
            self,
            dims_in: int,
            dims_out: int,
            depth_scheme: list[int],
            width_scheme: list[int],
            batch_norm: bool = True,
            activation: nn.Module = nn.ReLU(),
    ):
        # check if the schemes have the same length
        if len(depth_scheme) != len(width_scheme):
            raise ValueError(f"depth_scheme ({len(depth_scheme)}) and width_scheme ({len(width_scheme)}) must have the "
                             "same length.")
        # create the network
        layers = [nn.Linear(dims_in, width_scheme[0])]
        for i in range(len(depth_scheme)):
            layers.append(ConstWidth(depth_scheme[i], width_scheme[i], residual=True, batch_norm=batch_norm,
                                     activation=activation))
            if i < len(depth_scheme) - 1:
                layers.append(activation)
                layers.append(nn.Linear(width_scheme[i], width_scheme[i + 1]))
        layers.append(nn.Linear(width_scheme[-1], dims_out))
        super().__init__(*layers)


class NormalizerFreeResNetHParams(SubnetHParams):
    subnet_class = "NormalizerFreeResNet"
    depth_scheme: list[int]
    width_scheme: list[int]
    alpha: float
    activation: nn.Module = nn.ReLU()
    scaled_weights: bool = True


class NormalizerFreeResNet(nn.Sequential):
    def __init__(
            self,
            dims_in: int,
            dims_out: int,
            depth_scheme: list[int],
            width_scheme: list[int],
            alpha: float,
            activation: nn.Module = nn.ReLU(),
            scaled_weights: bool = True,
    ):
        # check if the schemes have the same length
        if len(depth_scheme) != len(width_scheme):
            raise ValueError(f"depth_scheme ({len(depth_scheme)}) and width_scheme ({len(width_scheme)}) must have the "
                             "same length.")
        # decide between scaled weights and normal weights linear layer
        if scaled_weights:
            Linear = ScaledWSLinear
            kwargs = {"previous_activation": activation}
        else:
            Linear = nn.Linear
            kwargs = {}
        layers = [Linear(dims_in, width_scheme[0])]
        # project to first width
        for i in range(len(depth_scheme)):
            layers.append(NormalizerFreeConstWidth(depth_scheme[i], width_scheme[i], alpha, activation=activation,
                                                   scaled_weights=scaled_weights))
            # transition block to new width
            layers.append(activation)
            if i < len(depth_scheme) - 1:
                layers.append(Linear(width_scheme[i], width_scheme[i + 1], **kwargs))
            # at the end, project to output dimensions
            else:
                layers.append(Linear(width_scheme[i], dims_out, **kwargs))
        super().__init__(*layers)
        # initialise the transition layers semi-orthogonally as well
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
