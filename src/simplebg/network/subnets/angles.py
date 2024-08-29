import torch
from torch import nn
import numpy as np

def angle_to_embedded(theta):
    """Convert an angle to an embedded representation."""
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

def embedded_to_angle(x):
    """Convert an embedded representation to an angle."""
    return torch.atan2(x[..., 1], x[..., 0])

def angle_to_embedded_flat(theta):
    """Convert an angle to an embedded representation."""
    x = angle_to_embedded(theta)
    return x.flatten(start_dim=-2)

def embedded_flat_to_angle(x):
    """Convert an embedded representation to an angle."""
    x = x.view(x.shape[:-1] + (2, -1))
    return embedded_to_angle(x)

class AngleShift(nn.Module):
    def __init__(self, dims_in, support=None):
        super().__init__()
        if support is None:
            support = (-np.pi, np.pi)
        self.dims_in = dims_in
        self.thetas = nn.Parameter(torch.zeros(dims_in))
        self.lower_bound = support[0]
        self.interval_length = support[1] - support[0]

    def encode(self, x):
        return (x + self.thetas) % self.interval_length + self.lower_bound

    def decode(self, x):
        return (x - self.thetas) % self.interval_length + self.lower_bound