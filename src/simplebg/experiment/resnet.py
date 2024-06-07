from . import core
from .. import latent
from ..network import resnet as resnetnn


# TODO: it does not make sense to have different experiments for different network types. Rework so network can
#  become a hyperparameter, too.

class PeptideHParams(core.PeptideHParams):
    network_hparams: resnetnn.ResNetHParams


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
        NN = getattr(resnetnn, self.hparams.network_hparams.network)
        self.nn = NN(dims_in, self.hparams.network_hparams)
        # create the latent distribution
        Q = getattr(latent, self.hparams.latent_hparams.name)
        self.q = Q(dims=self.nn.dims_out, **self.hparams.latent_hparams.kwargs)
