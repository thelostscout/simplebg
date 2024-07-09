from torch import nn

from lightning_trainable.hparams import HParams

from .buildingblocks import ResidualBlock


class SubnetHParams(HParams):
    subnet_class: str


class FullyConnectedHParams(SubnetHParams):
    subnet_class = "FullyConnected"
    width: int
    activation: nn.Module = nn.ReLU()
    batch_norm: bool = False


class FullyConnected(nn.Sequential):
    def __init__(
            self,
            dims_in: int,
            dims_out: int,
            width: int,
            activation: nn.Module = nn.ReLU(),
            batch_norm: bool = False,
    ):
        layers = [nn.Linear(dims_in, width),
                  ResidualBlock(width, depth=1, residual=False, batch_norm=batch_norm, activation=activation),
                  nn.Linear(width, dims_out)]
        super().__init__(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
