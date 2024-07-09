import torch

import lightning_trainable as lt
from lightning_trainable.hparams import AttributeDict
from FrEIA.utils import force_to

from ..data import loaders
from .. import network
from ..loss import core as loss
from .. import latent, evaluate


class BaseHParams(lt.TrainableHParams):
    model_class: str
    loader_hparams: loaders.LoaderHParams
    network_hparams: network.NetworkHParams
    latent_hparams: latent.DistributionHParams
    loss_weights: loss.LossWeights

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        nn_module = getattr(network, hparams.network_hparams.network_module)
        NN = getattr(nn_module, hparams.network_hparams.network_class)
        if NN.exact_invertible:
            if hparams.loss_weights.reconstruction:
                raise ValueError("Your network is exactly invertible and doesn't need a reconstruction loss.")
        else:
            if not hparams.loss_weights.reconstruction:
                raise ValueError("Your network is not exactly invertible and therefore needs a reconstruction loss.")
        return hparams


class BaseModel(lt.trainable.Trainable):
    hparams_type = BaseHParams
    hparams: BaseHParams
    loader_class = loaders.Loader

    def __init__(
            self,
            hparams: BaseHParams | dict
    ):
        super().__init__(hparams)
        self.loader = self.loader_class(self.hparams.loader_hparams)
        self.train_data, self.val_data, self.test_data = self.loader.generate_datasets()

    def compute_metrics(self, batch, batch_idx) -> dict:
        kwargs = dict()
        if "reverse_kl" in self.hparams.loss_weights.active_to_dict().keys():
            kwargs["energy_function"] = self.loader.energy_function
        # TODO: this is still ugly, maybe find something better?
        if type(batch) is list:
            batch = batch[0]
        progress = self.current_epoch / self.hparams.max_epochs
        total_loss, loss_dict = loss.compute_losses(batch, self.nn, self.q, self.hparams.loss_weights, progress, **kwargs)
        return dict(loss=total_loss, **loss_dict)

    def log_prob(self, x):
        z, log_det_jf, _ = self.nn.forward(x, jac=True)
        return log_det_jf + self.q.log_prob(z)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            z = self.q.sample(sample_shape)
            x = self.nn.reverse(z, jac=False)[0]
        return x

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
    loader_hparams: loaders.PeptideLoaderHParams


class PeptideModel(BaseModel):
    hparams_type = PeptideHParams
    hparams: PeptideHParams
    loader_class = loaders.PeptideLoader

    def __init__(
            self,
            hparams: PeptideHParams | dict
    ):
        super().__init__(hparams)
        # the input dimensions for the network are determined by the data
        nn_module = getattr(network, self.hparams.network_hparams.network_module)
        NN = getattr(nn_module, self.hparams.network_hparams.network_class)
        self.nn = NN(self.loader.dims, self.hparams.network_hparams, system=self.loader.system)
        Q = getattr(latent, self.hparams.latent_hparams.name)
        self.q = Q(dims=self.nn.dims_out, **self.hparams.latent_hparams.kwargs)

    def compute_metrics(self, batch, batch_idx) -> dict:
        metrics = super().compute_metrics(batch, batch_idx)
        if not self.training:
            energy = evaluate.sample_energies(self, n_samples=batch[0].shape[0]//10)
            metrics["mean_energy"] = energy.mean()
        return metrics

    @property
    def peptide(self):
        return self.loader


class ToyHParams(BaseHParams):
    model_class = "ToyModel"
    loader_hparams: loaders.ToyLoaderHParams


class ToyModel(BaseModel):
    hparams_type = ToyHParams
    hparams: ToyHParams
    loader_class = loaders.ToyLoader

    def __init__(
            self,
            hparams: ToyHParams | dict
    ):
        super().__init__(hparams)
        # the input dimensions for the network are determined by the data
        nn_module = getattr(network, self.hparams.network_hparams.network_module)
        NN = getattr(nn_module, self.hparams.network_hparams.network_class)
        self.nn = NN(self.loader.dims, self.hparams.network_hparams)
        Q = getattr(latent, self.hparams.latent_hparams.name)
        self.q = Q(dims=self.nn.dims_out, **self.hparams.latent_hparams.kwargs)

    @property
    def toy(self):
        return self.loader
