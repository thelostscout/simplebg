# Adapted from https://github.com/vislearn/FFF/blob/main/fff/loss.py
#
# MIT License
#
# Copyright (c) 2024 Computer Vision and Learning Lab, Heidelberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools
from collections import namedtuple
from functools import wraps, partial
from math import sqrt, prod
from typing import Union, Callable

import torch
from torch import vmap
from torch._functorch.eager_transforms import jacrev, jacfwd
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

from lightning_trainable.hparams import AttributeDict

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "surrogate", "regularizations"])
ExactOutput = namedtuple("ExactOutput", ["z", "x1", "nll"])
Transform = Callable[[torch.Tensor], torch.Tensor]


def nll_fff(
        metrics: AttributeDict,
        **kwargs
):
    if kwargs["training"]:
        z, x1, loss, *_ = nll_surrogate(metrics, **kwargs)
    else:
        z, x1, loss = nll_exact(metrics, **kwargs)
    return loss


def nll_surrogate(
        metrics: AttributeDict,
        **kwargs
) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.
    """
    hutchinson_samples = kwargs.get("hutchinson_samples", 1)
    decode = kwargs["nn"].reverse

    surrogate = 0
    vs = sample_v(metrics.z, hutchinson_samples)
    for k in range(hutchinson_samples):
        v = vs[..., k]

        # $ g'(z) v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(metrics.z, v)
            dual_x1 = decode(dual_z)
            x1, v1 = unpack_dual(dual_x1)

        # $ v^T f'(x) $ via backward-mode AD
        v2, = grad(metrics.z, metrics.x, v, create_graph=True)

        # $ v^T f'(x) stop_grad(g'(z)) v $
        surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples

    # Per-sample negative log-likelihood
    nll = sum_except_batch((metrics.z ** 2)) / 2 - surrogate

    return SurrogateOutput(metrics.z, x1, nll, surrogate, {})


def nll_exact(metrics, **kwargs) -> ExactOutput:
    """
    Compute the nll of the decoder.

    It will return a lower bound (unless the decoder is invertible)
    of the probability density the decoder assigns to the points
    x1 = decode(encode(x)) -- if the reconstruction is good, then this
    is a good approximation of the nll by the dataset.
   """
    decode = kwargs["nn"].reverse
    latent_distribution = kwargs["latent_distribution"]

    x1, dec_jac = compute_jacobian(metrics.z, decode)
    return ExactOutput(metrics.z, x1, -latent_distribution.log_prob(metrics.z) + torch.slogdet(dec_jac).logabsdet)


# def fff_loss(x: torch.Tensor,
#              encode: Transform, decode: Transform,
#              beta: Union[float, torch.Tensor],
#              hutchinson_samples: int = 1) -> torch.Tensor:
#     """
#     Compute the per-sample FFF/FIF loss:
#     $$
#     \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
#     $$
#     where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of `encode` and `decode`.
#
#     :param x: Input data. Shape: (batch_size, ...)
#     :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
#     :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
#     :param beta: Weight of the mean squared error.
#     :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
#     :return: Per-sample loss. Shape: (batch_size,)
#     """
#     surrogate = nll_surrogate(x)
#     mse = torch.sum((x - surrogate.x1) ** 2, dim=tuple(range(1, len(x.shape))))
#     return beta * mse + surrogate.nll


def sample_v(x: torch.Tensor, hutchinson_samples: int):
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data.
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :return:
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])
    if hutchinson_samples > total_dim:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {total_dim}")
    v = torch.randn(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
    return q * sqrt(total_dim)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sum over all dimensions except the first.
    :param x: Input tensor.
    :return: Sum over all dimensions except the first.
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)


def batch_wrap(fn):
    """
    Add a batch dimension to each tensor argument.

    :param fn:
    :return:
    """

    def deep_unsqueeze(arg):
        if torch.is_tensor(arg):
            return arg[None, ...]
        elif isinstance(arg, dict):
            return {key: deep_unsqueeze(value) for key, value in arg.items()}
        elif isinstance(arg, (list, tuple)):
            return [deep_unsqueeze(value) for value in arg]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = deep_unsqueeze(args)
        return fn(*args, **kwargs)[0]

    return wrapper


def double_output(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out

    return wrapper


def compute_jacobian(x_in, fn, *func_args, chunk_size=None, grad_type="backward", **func_kwargs):
    jacfn = jacrev if grad_type == "backward" else jacfwd
    with torch.inference_mode(False):
        with torch.no_grad():
            fn_kwargs_prefilled = partial(fn, **func_kwargs)
            fn_batch_expanded = batch_wrap(fn_kwargs_prefilled)
            fn_return_val = double_output(fn_batch_expanded)
            fn_jac_batched = vmap(
                jacfn(fn_return_val, has_aux=True), chunk_size=chunk_size
            )
            jac, x_out = fn_jac_batched(x_in, *func_args)
    return x_out, jac
