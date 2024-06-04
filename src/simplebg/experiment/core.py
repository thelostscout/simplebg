import torch

import lightning_trainable as lt
from FrEIA.utils import force_to

from ..data import loader
from ..network import core as network
from ..loss import core as loss
from .. import latent


class BaseHParams(lt.TrainableHParams):
    loader_hparams: loader.LoaderHParams
    network_hparams: network.NetworkHParams
    latent_hparams: latent.DistributionHParams
    loss_weights: loss.LossWeights


class BaseExperiment(lt.trainable.Trainable):
    hparams_type = BaseHParams
    hparams: BaseHParams

    def load_data(self):
        """
        Load the data according to the loader_hparams.
        :return:
        train_data, val_data, test_data
        """
        raise NotImplementedError

    def compute_metrics(self, batch, batch_idx) -> dict:
        losses, loss_dict = loss.compute_losses_single(batch, self.nn, self.q, self.hparams.loss_weights,
                                                       training=self.training)
        return dict(loss=losses, **loss_dict)

    def log_prob(self, x):
        z, log_det_jf = self.nn.forward(x)
        return log_det_jf + self.q.log_prob(z)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            z = self.q.sample(sample_shape)
            x = self.nn.reverse(z)[0]
        return x

    def __init__(
            self,
            hparams: BaseHParams | dict
    ):
        super().__init__(hparams)
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.nn = None
        self.q = None

    def to(self, *args, **kwargs):
        """
        Wrapper function to move the parameters of the latent distribution between devices.
        This is neccesary because torch distributions can't move their parameters between devices by default.
        """
        #
        force_to(self.q, *args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        # Delegate to .to(...) as it moves distributions, too.
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        return self.to(device)

    def cpu(self):
        # Delegate to .to(...) as it moves distributions, too.
        return self.to("cpu")


class PeptideHParams(BaseHParams):
    loader_hparams: loader.PeptideLoaderHParams


class PeptideExperiment(BaseExperiment):
    hparams_type = PeptideHParams
    hparams: PeptideHParams

    def __init__(
            self,
            hparams: PeptideHParams | dict
    ):
        self.peptide = None
        super().__init__(hparams)

    def load_data(self):
        self.peptide = loader.PeptideLoader(self.hparams.loader_hparams)
        return self.peptide.generate_datasets()

    def compute_metrics(self, batch, batch_idx) -> dict:
        x = batch[0]
        losses, loss_dict = loss.compute_losses(x, self.nn, self.q, self.hparams.loss_weights,
                                                testing=self.trainer.testing, validating=self.trainer.validating)
        return dict(loss=losses, **loss_dict)


class ToyHParams(BaseHParams):
    loader_hparams: loader.ToyLoaderHParams


class ToyExperiment(BaseExperiment):
    hparams_type = ToyHParams
    hparams: ToyHParams

    def __init__(
            self,
            hparams: ToyHParams | dict
    ):
        self.toy = None
        super().__init__(hparams)

    def load_data(self):
        self.toy = loader.ToyLoader(self.hparams.loader_hparams)
        return self.toy.generate_datasets()
