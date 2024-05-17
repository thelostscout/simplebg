from torch import nn
import math
import warnings

from lightning_trainable.hparams import HParams


class SubnetHParams(HParams):
    """This base class is empty, but it serves as a type hint for all other subnet hparams."""

class LinearSubnetHParams(SubnetHParams):
    depth: int
    width: int = 128


class LinearSubnet(nn.Sequential):
    def __init__(self, dims_in: int, dims_out: int, depth: int, width: int):
        # sanity check
        if dims_in > width or dims_out > width:
            raise ValueError(f"dims_in ({dims_in}) or dims_out ({dims_out}) is greater than width ({width}).")
        # create the network
        layers = [nn.Linear(dims_in, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        # last layer is special because dims_out is fixed, and we don't have an activation
        layers.append(nn.Linear(width, dims_out))
        super().__init__(*layers)
        self[-1].weight.data.zero_()
        self[-1].bias.data.zero_()


class ExponentialSubnetHParams(SubnetHParams):
    depth: int
    max_width: int = 512
    growth: float = 2.0


class ExponentialSubnet(nn.Sequential):
    def __init__(self, dims_in: int, dims_out: int, depth: int, max_width: int = 512, growth: float = 2.0):
        # sanity check
        if dims_in > max_width or dims_out > max_width:
            raise ValueError(f"dims_in ({dims_in}) or dims_out ({dims_out}) is greater than max_width ({max_width}).")
        # find turning point: shift from the middle into the direction where there are more dimensions
        turning_point = depth / 2 + math.log(dims_out / dims_in) / math.log(growth) / 2
        # check and warn if the turning point is out of bounds. Warn in this case that the layer sizes will only grow
        # or shrink
        if turning_point > depth:
            warnings.warn("The relationship between dims_in, dims_out and growth leads to exponentially growing layer "
                          "sizes only instead of shrinking again in the end.")
            turning_point = depth
        elif turning_point < 0:
            warnings.warn("The relationship between dims_in, dims_out and growth leads to exponentially shrinking "
                          "layer sizes only instead of growing first.")
            turning_point = 0
        # calculate the exponentially growing and shrinking layer sizes
        layer_sizes = []
        width = dims_in
        for i in range(depth):
            # grow width up to turning point
            if i < turning_point:
                new_width = width * growth
            # shrink after
            else:
                new_width = width / growth
            layer_sizes.append((width, new_width))
            width = new_width
        # last layer is special case because dims_out is fixed
        layer_sizes.append((width, dims_out))
        # create the network
        layers = []
        for i, size in enumerate(layer_sizes):
            # round floats to integers and truncate layer sizes to width
            int_size = (min(round(size[0]), max_width), min(round(size[1]), max_width))
            layers.append(nn.Linear(*int_size))
            # all but the last layer receive a ReLU activation
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
        super().__init__(*layers)
        # initialize the last layer to zero because it helps the training (somehow)
        self[-1].weight.data.zero_()
        self[-1].bias.data.zero_()
