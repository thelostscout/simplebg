import torch
from lightning_trainable.hparams import HParams, AttributeDict
from .. import network, latent
from . import fff, kl, misc


class LossWeights(HParams):
    forward_kl: float = 0.
    nll_surrogate: float = 0.
    reconstruction: float = 0.

    def active_to_dict(self):
        return {k: v for k, v in self.items() if v > 0}


function_metrics = dict(
    # each function should have a set of required metrics
    forward_kl={"forward_pass", "forward_jac", },
    reverse_kl={"generated_samples", "samples_jac", },
    nll_surrogate={"forward_pass", },
    reconstruction={"forward_pass", "reverse_pass", },
)

function_map = dict(
    forward_kl=kl.forward_kl,
    reverse_kl=kl.reverse_kl,
    nll_surrogate=fff.nll_surrogate,
    reconstruction=misc.reconstruction,
)


def loss(
        x: torch.Tensor,
        nn: network.base.BaseNetwork,
        latent_distribution: latent.BaseDistribution,
        loss_weights: LossWeights,
        evaluating: bool = False,
        testing: bool = False,
        **kwargs,
):
    active = loss_weights.active_to_dict().keys()
    required_metrics = set.union(*[function_metrics[function] for function in active])
    metrics = calculate_metrics(required_metrics, x, nn, latent_distribution)
    loss_dict = AttributeDict()
    fn_kwargs = dict(
        latent_distribution=latent_distribution,
        nn=nn,
        evaluating=evaluating,
        testing=testing,
        **kwargs,
    )
    for function in active:
        loss_dict[function] = function_map[function](metrics, **fn_kwargs)


# TODO: some loss functions calculate some of the metrics. Maybe find a clever way to check which calculate which
#  metrics and how to apply those first.

def calculate_metrics(
        required_metrics: set[str],
        x: torch.Tensor,
        nn: network.base.BaseNetwork,
        latent_distribution: latent.BaseDistribution
):
    """The calculation of metrics needs to be hardcoded to ensure an efficient order of operations."""
    metrics = AttributeDict()
    if "forward_pass" in required_metrics:
        z, log_det_j_forward = forward_pass(x, nn)
        metrics.z = z
        # The network might not compute the jacobian, which is no inherent problem. But if we want to use it,
        # we need to make sure that the network actually did compute it.
        if "forward_jac" in required_metrics:
            if log_det_j_forward is None:
                raise ValueError(
                    "log_det_j_forward is required, but was not computed. This is probably because your network does "
                    "not support jacobian computation.")
            metrics.log_det_j_forward = log_det_j_forward
        # a reverse pass only makes sense if there was a forward pass and the network is not bijective. In other
        # cases we generate samples first and then pass them through the network reversely (see "generated_samples"
        # below).
        if "reverse_pass" in required_metrics:
            x1, log_det_j_reverse = reverse_pass(z, nn)
            metrics.x1 = x1
            # The network might not compute the jacobian, which is no inherent problem. But if we want to use it,
            # we need to make sure that the network actually did compute it.
            if "reverse_jac" in required_metrics:
                if log_det_j_reverse is None:
                    raise ValueError(
                        "log_det_j_reverse is required, but was not computed. This is probably because your network "
                        "does not support jacobian computation.")
                metrics.log_det_j_reverse = log_det_j_reverse
    if "generated_samples" in required_metrics:
        x_generated, log_det_j_samples = generate_samples(x, latent_distribution, nn)
        metrics.x_generated = x_generated
        # The network might not compute the jacobian, which is no inherent problem. But if we want to use it,
        # we need to make sure that the network actually did compute it.
        if "samples_jac" in required_metrics:
            if log_det_j_samples is None:
                raise ValueError(
                    "log_det_j_samples is required, but was not computed. This is probably because your network does "
                    "not support jacobian computation.")
            metrics.log_det_j_samples = log_det_j_samples
    return metrics


def forward_pass(
        x: torch.Tensor,
        nn: network.base.BaseNetwork,
):
    return nn.forward(x)


def reverse_pass(
        z: torch.Tensor,
        nn: network.base.BaseNetwork,
):
    return nn.reverse(z)


def generate_samples(
        x: torch.Tensor,
        latent_distribution: latent.BaseDistribution,
        nn: network.base.BaseNetwork,
):
    sample_shape = x.shape[:-1]
    z_generated = latent_distribution.sample(sample_shape)
    return reverse_pass(z_generated, nn)
