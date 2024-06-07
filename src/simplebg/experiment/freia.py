from . import core
from .. import latent
from ..network import freia as freiann


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
