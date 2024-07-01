import torch
from torch import nn


class ScaledWSLinear(nn.Linear):
    """
    Linear layer with Scaled Weight Standardization.
    Implementation adapted from Conv2D implementation with scaled weights in arXiv:2101.08692
    """
    @staticmethod
    def calculate_gain(activation, size):
        y = activation(torch.randn(size))
        return torch.sqrt(torch.mean(torch.var(y, dim=1)))

    def __init__(self, in_features, out_features, bias=True, previous_activation=None, eps=1e-4, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        if previous_activation is not None:
            self.gain = nn.Parameter(self.calculate_gain(previous_activation, (1024, 256)))  # shape is arbitrary
        else:
            self.gain = None
        # Epsilon, a small constant to avoid dividing by zero.
        self.eps = eps

    def get_weight(self):
        # Get Scaled WS weight OIHW;
        weight = ((self.weight - self.weight.mean(dim=1)) /
                  torch.sqrt(self.weight.var(dim=1) * self.weight.shape[1] + self.eps))
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return nn.functional.linear(x, self.get_weight(), self.bias)

class ResidualBlock(nn.Sequential):
    def __init__(
            self,
            width: int,
            depth: int = 1,
            residual: bool = False,
            batch_norm: bool = False,
            activation: nn.Module = nn.ReLU,
    ):
        self.residual = residual
        layers = []
        for i in range(depth):
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation())
            layers.append(nn.Linear(width, width))
        super().__init__(*layers)
        # initialize weights and biases according to arXiv:1712.05577
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        if self.residual:
            return input + super().forward(input)
        else:
            return super().forward(input)


class ConstWidth(nn.Sequential):
    def __init__(
            self,
            depth: int,
            width: int,
            block_depth: int = 2,
            residual: bool = False,
            batch_norm: bool = False,
            activation: nn.Module = nn.ReLU,
    ):
        layers = []
        # add residual blocks
        for i in range(depth):
            layers.append(
                ResidualBlock(width, depth=block_depth, residual=residual, batch_norm=batch_norm,
                              activation=activation))
        super().__init__(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)


class NormalizerFreeResidualBlock(nn.Sequential):
    """
    Implementation of a normalizer-free residual block with Scaled Weight Standardization.
    Follows the archictecture implemented in arXiv:2101.08692
    """
    def __init__(
            self,
            width: int,
            alpha: float,
            beta: float,
            depth: int = 1,
            activation: nn.Module = nn.ReLU,
    ):
        self.alpha = alpha
        self.beta = beta
        layers = []
        for i in range(depth):
            layers.append(activation())
            layers.append(ScaledWSLinear(width, width, bias=True, previous_activation=activation))
        super().__init__(*layers)
        # initialize weights and biases according to arXiv:1712.05577
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        return input + self.alpha * super().forward(input / self.beta)


class NormalizerFreeConstWidth(nn.Sequential):
    def __init__(
            self,
            depth: int,
            width: int,
            alpha: float,
            block_depth: int = 2,
            activation: nn.Module = nn.ReLU,
    ):
        layers = []
        expected_var = 1.
        # add residual blocks
        for i in range(depth):
            layers.append(
                NormalizerFreeResidualBlock(width, alpha, beta=expected_var**.5, depth=block_depth,
                                            activation=activation))
            expected_var += alpha ** 2
        super().__init__(*layers)
