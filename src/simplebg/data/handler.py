from lightning_trainable.hparams import HParams
from lightning_trainable.hparams.types import Choice

from .dataset import *
from .loader import PeptideLoaderHParams, PeptideLoader
from ..transforms.ic import CartesianToInternalTransform


class PetideHandlerHParams(HParams):
    representation: Choice("cartesian_coordinates", "internal_coordinates")
    load_params: PeptideLoaderHParams


class PeptideHandler:
    hparams: PetideHandlerHParams

    @property
    def name(self):
        return self.hparams.load_params.name

    @property
    def system(self):
        return self._system

    @property
    def temperature(self):
        return self._temperature

    @property
    def dataset(self):
        return self._dataset


    def __init__(self, hparams):
        self.hparams = hparams
        dataloader = PeptideLoader(hparams.load_params)
        system, xyz, temperature = dataloader.load()
        # need to reinitialize the energy model to set n_workers to 1 due to a bug:
        # https://github.com/noegroup/bgflow/issues/35
        system = system.reinitialize_energy_model(temperature=temperature, n_workers=1)
        self._system = system
        self._temperature = temperature
        dataset = PeptideCCDataset(xyz)
        if hparams.representation == "cartesian_coordinates":
            self._dataset = dataset
        else:
            self._dataset = CartesianToInternalTransform(system).forward(dataset)[0]