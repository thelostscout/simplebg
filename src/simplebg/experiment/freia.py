from . import core
from ..network import freia as freiann
from ..loss.core import compute_losses
from .. import latent


class ToyHParams(core.ToyHParams):
    network_hparams: freiann.FrEIABaseHParams


class ToyExperiment(core.ToyExperiment):
    hparams_type = core.ToyHParams
    hparams: core.ToyHParams

    def __init__(
            self,
            hparams: core.ToyHParams | dict
    ):
        super().__init__(hparams)
        # the input dimensions for the network are determined by the data
        dims_in = self.toy.dims
        NN = getattr(freiann, self.hparams.network_hparams.network)
        self.nn = NN(dims_in, self.hparams.network_hparams)
        Q = getattr(latent, self.hparams.latent_hparams.name)
        self.q = Q(dims=self.nn.dims_out, **self.hparams.latent_hparams.kwargs)

    # def compute_metrics(self, batch, batch_idx) -> dict:
    #     x = batch
    #     z, log_det_j = self.nn.forward(x)
    #     loss = - self.q.log_prob(z) - log_det_j
    #     return dict(loss=loss.mean())


class PeptideHParams(core.PeptideHParams):
    network_hparams: freiann.FrEIABaseHParams


class PeptideExperiment(core.PeptideExperiment):
    hparams_type = core.PeptideHParams
    hparams: core.PeptideHParams

    def __init__(
            self,
            hparams: core.PeptideHParams | dict
    ):
        super().__init__(hparams)
        # create the network
        # the input dimensions for the network are determined by the data
        dims_in = self.peptide.dims
        NN = getattr(freiann, self.hparams.network_hparams.network)
        self.nn = NN(dims_in, self.hparams.network_hparams)
        # create the latent distribution
        Q = getattr(latent, self.hparams.latent_hparams.name)
        self.q = Q(dims=self.nn.dims_out, **self.hparams.latent_hparams.kwargs)

    def compute_metrics(self, batch, batch_idx) -> dict:
        x = batch[0]
        z, log_det_j = self.nn.forward(x)
        loss = - self.q.log_prob(z) - log_det_j
        return dict(loss=loss.mean())
