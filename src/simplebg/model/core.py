import torch

import lightning_trainable as lt
from FrEIA.utils import force_to

from ..data import loader
from .. import network
from ..loss import core as loss
from .. import latent


class BaseHParams(lt.TrainableHParams):
    model_class: str
    loader_hparams: loader.LoaderHParams
    network_hparams: network.NetworkHParams
    latent_hparams: latent.DistributionHParams
    loss_weights: loss.LossWeights


class BaseModel(lt.trainable.Trainable):
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
        z, log_det_jf = self.nn.forward(x, jac=True)
        return log_det_jf + self.q.log_prob(z)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            z = self.q.sample(sample_shape)
            x = self.nn.reverse(z)[0]
        return x

    @property
    def data_dims(self):
        raise NotImplementedError

    def __init__(
            self,
            hparams: BaseHParams | dict
    ):
        super().__init__(hparams)
        self.train_data, self.val_data, self.test_data = self.load_data()
        # the input dimensions for the network are determined by the data
        nn_module = getattr(network, self.hparams.network_hparams.network_module)
        NN = getattr(nn_module, self.hparams.network_hparams.network_class)
        self.nn = NN(self.data_dims, self.hparams.network_hparams)
        Q = getattr(latent, self.hparams.latent_hparams.name)
        self.q = Q(dims=self.nn.dims_out, **self.hparams.latent_hparams.kwargs)

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
    model_class = "PeptideModel"
    loader_hparams: loader.PeptideLoaderHParams


class PeptideModel(BaseModel):
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
        losses, loss_dict = loss.compute_losses_single(x, self.nn, self.q, self.hparams.loss_weights,
                                                       training=self.training)
        return dict(loss=losses, **loss_dict)

    @property
    def data_dims(self):
        return self.peptide.dims


class ToyHParams(BaseHParams):
    model_class = "ToyModel"
    loader_hparams: loader.ToyLoaderHParams


class ToyModel(BaseModel):
    hparams_type = ToyHParams
    hparams: ToyHParams

    def __init__(
            self,
            hparams: ToyHParams | dict
    ):
        self.toy = None
        super().__init__(hparams)

    @property
    def data_dims(self):
        return self.toy.dims

    def load_data(self):
        self.toy = loader.ToyLoader(self.hparams.loader_hparams)
        return self.toy.generate_datasets()
