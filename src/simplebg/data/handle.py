from lightning_trainable.hparams import HParams
from lightning_trainable.hparams.types import Choice

from .dataset import BaseDataset
from .load import PeptideLoaderHParams, PeptideLoader


class PetideHandlerHParams(HParams):
    representation: Choice("cartesian", "internal")
    load_params: PeptideLoaderHParams


class PeptideHandler:
    hparams: PetideHandlerHParams

    def __init__(self, hparams):
        dataloader = PeptideLoader(hparams.load_params)
        system, xyz, temperature = dataloader.load()
        # need to reinitialize the energy model to set n_workers to 1 due to a bug:
        # https://github.com/noegroup/bgflow/issues/35
        system = system.reinitialize_energy_model(temperature=temperature, n_workers=1)
        self.system = system  # is this a neccecary attribute?
        if hparams.representation == "cartesian":
            self.dataset = BaseDataset(xyz)
