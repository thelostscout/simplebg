import torch
from lightning_trainable.hparams import HParams, AttributeDict
from .. import latent
from ..network import core as network
from . import fff, kl, misc


class LossWeights(HParams):
    forward_kl: float = 0.
    reverse_kl: float = 0.
    fff: float = 0.
    reconstruction: float = 0.

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        if not [v for v in hparams.values() if v != 0]:
            raise ValueError("At least one loss function must be active.")
        return hparams

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
    nll_surrogate=fff.general_loss,
    reconstruction=misc.reconstruction,
)


def compute_losses_single(
        x: torch.Tensor,
        nn: network.BaseNetwork,
        latent_distribution: latent.BaseDistribution,
        loss_weights: LossWeights,
        training: bool,
        **kwargs,
):
    active = loss_weights.active_to_dict().keys()
    if len(active) > 1:
        raise ValueError("Naive loss computation only supports one active loss function at a time.")
    metrics = AttributeDict()
    if active == {"forward_kl"}:
        metrics.z, metrics.log_det_j_forward = forward_pass(x, nn)
        forward_kl = kl.forward_kl(metrics, latent_distribution=latent_distribution).mean()
        return forward_kl, {"forward_kl": forward_kl}
    elif active == {"reverse_kl"}:
        metrics.x_generated, metrics.log_det_j_samples = generate_samples(x, latent_distribution, nn)
        reverse_kl = kl.reverse_kl(metrics, target_energy=latent_distribution.log_prob).mean()
        return reverse_kl, {"reverse_kl": reverse_kl}
    elif active == {"fff"}:
        metrics.z, metrics.log_det_j_forward = forward_pass(x, nn)
        beta = 1_000
        fff_loss = fff.general_loss(
            x=x,
            encode=lambda y: nn.forward(y)[0],
            decode=lambda y: nn.reverse(y)[0],
            latent_distribution=latent_distribution,
            beta=beta,
            training=training,
            **kwargs,
        )
        return fff_loss.loss.mean(), {"nll": fff_loss.nll.mean(), "reconstruction": fff_loss.mse.mean()}
    elif active == {"reconstruction"}:
        metrics.z, metrics.log_det_j_forward = forward_pass(x, nn)
        metrics.x1, metrics.log_det_j_reverse = reverse_pass(metrics.z, nn)
        reconstruction = misc.reconstruction(metrics).mean()
        return reconstruction, {"reconstruction": reconstruction}


def compute_losses(
        x: torch.Tensor,
        nn: network.BaseNetwork,
        latent_distribution: latent.BaseDistribution,
        loss_weights: LossWeights,
        training: bool,
        **kwargs,
):
    raise NotImplementedError("This implementation of this function currently doesn't work. It is in active "
                              "development.")
    active = loss_weights.active_to_dict().keys()
    loss = 0.
    metrics = AttributeDict()
    z = None
    log_det_j_forward = None
    x1 = None
    log_det_j_reverse = None
    x_generated = None
    log_det_j_samples = None
    if "fff" in active:
        if "reconstruction" in active:
            beta = loss_weights.rec / loss_weights.fff
            print(f"calculating beta={beta:.1f}=reconstruction/fff")
        else:
            raise ValueError("The fff loss requires a weight for the reconstruction loss to calculate beta.")
        fff_loss = fff.general_loss(
            x=x,
            encode=lambda y: nn.forward(y)[0],
            decode=lambda y: nn.reverse(y)[0],
            latent_distribution=latent_distribution,
            beta=beta,
            training=training,
            **kwargs,
        )
        loss += fff_loss.loss.mean()
        metrics.nll = fff_loss.nll.mean()
        metrics.mse = fff_loss.mse.mean()
        z = fff_loss.z
        x1 = fff_loss.x1





# TODO: some loss functions calculate some of the metrics. Maybe find a clever way to check which calculate which
#  metrics and how to apply those first.

def calculate_metrics(
        required_metrics: set[str],
        x: torch.Tensor,
        nn: network.BaseNetwork,
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
                    "log_det_j_forward is required, but was not computed. This is probably because your network "
                    "does not support jacobian computation.")
            metrics.log_det_j_forward = log_det_j_forward
        # a reverse pass only makes sense if there was a forward pass and the network is not bijective. In other
        # cases we generate samples first and then pass them through the network reversely
        # (see "generated_samples" below).
        if "reverse_pass" in required_metrics:
            x1, log_det_j_reverse = reverse_pass(z, nn)
            metrics.x1 = x1
            # The network might not compute the jacobian, which is no inherent problem. But if we want to use it,
            # we need to make sure that the network actually did compute it.
            if "reverse_jac" in required_metrics:
                if log_det_j_reverse is None:
                    raise ValueError(
                        "log_det_j_reverse is required, but was not computed. This is probably because your "
                        "network does not support jacobian computation.")
                metrics.log_det_j_reverse = log_det_j_reverse
    if "generated_samples" in required_metrics:
        x_generated, log_det_j_samples = generate_samples(x, latent_distribution, nn)
        metrics.x_generated = x_generated
        # The network might not compute the jacobian, which is no inherent problem. But if we want to use it,
        # we need to make sure that the network actually did compute it.
        if "samples_jac" in required_metrics:
            if log_det_j_samples is None:
                raise ValueError(
                    "log_det_j_samples is required, but was not computed. This is probably because your network "
                    "does not support jacobian computation.")
            metrics.log_det_j_samples = log_det_j_samples
    return metrics


def forward_pass(
        x: torch.Tensor,
        nn: network.BaseNetwork,
):
    return nn.forward(x)


def reverse_pass(
        z: torch.Tensor,
        nn: network.BaseNetwork,
):
    return nn.reverse(z)


def generate_samples(
        x: torch.Tensor,
        latent_distribution: latent.BaseDistribution,
        nn: network.BaseNetwork,
):
    sample_shape = x.shape[:-1]
    z_generated = latent_distribution.sample(sample_shape)
    return reverse_pass(z_generated, nn)
