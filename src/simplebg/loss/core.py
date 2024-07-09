import torch
from lightning_trainable.hparams import HParams, AttributeDict
from .. import latent
from ..network import core as network
from . import fff, kl


class LossWeights(HParams):
    forward_kl: float = 0.
    reverse_kl: float = 0.
    reconstruction: float = 0.

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        if not [v for v in hparams.values() if v != 0]:
            raise ValueError("At least one loss function must be active.")
        return hparams

    def active_to_dict(self):
        return {k: v for k, v in self.items() if v > 0}


def compute_losses(
        x: torch.Tensor,
        nn: network.BaseNetwork,
        latent_distribution: latent.BaseDistribution,
        loss_weights: LossWeights,
        progress: float,
        nn_kwargs=None,
        **kwargs,
):
    if nn_kwargs is None:
        nn_kwargs = dict()
    active = loss_weights.active_to_dict().keys()
    loss_dict = dict()
    forward_pass = None
    if "forward_kl" in active:
        forward_pass = nn.forward(x, jac=True, **nn_kwargs)
        forward_kl = kl.forward_kl(forward_pass.output, forward_pass.log_det_j, latent_distribution).mean()
        loss_dict["forward_kl"] = forward_kl
    if "reverse_kl" in active:
        if progress > .5:
            z_generated = latent_distribution.sample((x.shape[0],))
            reverse_pass = nn.reverse(z_generated, jac=True, **nn_kwargs)
            reverse_kl = kl.reverse_kl(reverse_pass.output, reverse_pass.log_det_j, kwargs["energy_function"]).mean()
            loss_dict["reverse_kl"] = reverse_kl
        else:
            loss_dict["reverse_kl"] = torch.tensor(0.).to(x.device)
    if "reconstruction" in active:
        if forward_pass:
            if forward_pass.byproducts:
                x1 = forward_pass.byproducts.x1
            else:
                x1 = nn.reverse(forward_pass.output, jac=False, **nn_kwargs).output
        else:
            z = nn.forward(x, jac=False, **nn_kwargs).output
            x1 = nn.reverse(z, jac=False, **nn_kwargs).output
        reconstruction = fff.reconstruction_loss(x, x1).mean()
        loss_dict["reconstruction"] = reconstruction
    total_loss = torch.nansum(torch.stack([v * loss_dict[k] for k, v in loss_weights.items() if v > 0]))
    return total_loss, loss_dict
