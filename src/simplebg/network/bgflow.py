import torch

import bgflow
import bgmol
from torch import Tensor

from . import core


def _assert_in_unit_interval(x):
    if (x > 1 + 1e-6).any() or (x < - 1e-6).any():
        raise ValueError(f'IncreaseMultiplicityFlow operates on [0,1] but input was {x}')


class CircularShiftFlow(bgflow.Flow):
    """A flow that shifts the position of torsional degrees of freedom.
    The input and output tensors are expected to be in [0,1].
    The output is a translated version of the input, respecting circulariry.
    Parameters
    ----------
    shift : Union[torch.Tensor, float]
        A tensor that defines the translation of the circular interval
    """

    def __init__(self, shift):
        super().__init__()
        self.register_buffer("_shift", torch.as_tensor(shift))

    def _forward(self, x, **kwargs):
        _assert_in_unit_interval(x)
        y = (x + self._shift) % 1
        dlogp = torch.zeros_like(x[..., [0]])
        return y, dlogp

    def _inverse(self, x, **kwargs):
        _assert_in_unit_interval(x)
        y = (x - self._shift) % 1
        dlogp = torch.zeros_like(x[..., [0]])
        return y, dlogp


class NetworkWrapper(core.BaseNetwork):
    def __init__(self, flow: bgflow.Flow, dims_in: int, dims_out: tuple[int, ...]):
        super().__init__()
        self._flow = flow
        self._dims_in = dims_in
        self._dims_out = dims_out

    @property
    def dims_in(self):
        return self._dims_in

    @property
    def dims_out(self):
        return self._dims_out

    def forward(self, x: Tensor, *args, jac=True, **kwargs) -> core.NetworkOutput:
        # bgflow has the forward and inverse designed backwards, i.e. from the perspective of the latent space
        out = self._flow._inverse(x, *args, **kwargs)
        return core.NetworkOutput(out[:-1], out[-1], {})

    def reverse(self, z: Tensor, *args, jac=True, **kwargs) -> core.NetworkOutput:
        # bgflow has the forward and inverse designed backwards, i.e. from the perspective of the latent space
        out = self._flow._forward(z, *args, **kwargs)
        return core.NetworkOutput(out[:-1], out[-1], {})
