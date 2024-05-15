import torch
import torch.distributions as D

import lightning_trainable as lt

from simplebg.data.loader import PeptideLoaderHParams, PeptideLoader, split_dataset, DataSplitHParams


class BaseHParams(lt.TrainableHParams):
    loader_hparams: lt.hparams.HParams
    split_hparams: DataSplitHParams
    network_hparams: lt.hparams.HParams


class PeptideHParams(BaseHParams):
    loader_hparams: PeptideLoaderHParams


class BaseTrainable(lt.trainable.Trainable):
    hparams_type = BaseHParams
    hparams: BaseHParams

    @property
    def q(self):
        raise NotImplementedError

    @property
    def nn(self):
        raise NotImplementedError

    def load_data(self):
        """
        Load the data according to the loader_hparams.
        :return:
        train_data, val_data, test_data
        """
        raise NotImplementedError

    def log_prob(self, x):
        z, log_det_jf = self.nn.forward(x)
        return log_det_jf + self.q.log_prob(z)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            z = self.q.sample(sample_shape)
            x = self.nn.inverse(z)[0]
        return x

    def __init__(
            self,
            hparams: BaseHParams | dict
    ):
        train_data, val_data, test_data = self.load_data()
        super().__init__(hparams, train_data=train_data, val_data=val_data, test_data=test_data)


class PeptideTrainable(BaseTrainable):
    hparams_type = PeptideHParams
    hparams: PeptideHParams

    def __init__(
            self,
            hparams: PeptideHParams | dict
    ):
        self.peptide = None
        super().__init__(hparams)

    def load_data(self):
        self.peptide = PeptideLoader(self.hparams.loader_hparams)
        return split_dataset(self.peptide.cartesian, self.hparams.split_hparams)
