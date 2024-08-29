import os
import random
import warnings

from torch import Tensor
import mdtraj as md
import yaml
from abc import ABC, abstractmethod
from collections import namedtuple

import bgmol
import lightning_trainable as lt
from lightning_trainable.hparams import HParams, AttributeDict
from lightning_trainable.hparams.types import Choice

from ..utils.peptide import Peptide
from .. utils.path import get_default_path
from .dataset import PeptideCCDataset

datasets = namedtuple("datasets", ["train", "val", "test"])


def load_from_bgmol(
        root: str,
        name: str,
):
    DataSetClass = getattr(bgmol.datasets, name)
    if not os.path.exists(root):
        os.makedirs(root)
    # extract filename from url and replace .tgz with .npy
    # TODO: this handling of path might cause issues with different operating systems. Should do a OS agnostic method
    path = os.path.join(root, DataSetClass.url.split("/")[-1].replace(".tgz", ".npy"))
    download = not os.path.exists(path)
    dataset = DataSetClass(root=root, download=download, read=True)

    xyz_as_tensor = Tensor(dataset.xyz).view(*dataset.xyz.shape[:-2], -1)

    return dataset.system, xyz_as_tensor, dataset.temperature


def load_from_h5(
        root: str,
        name: str,
):
    # the topology, the trajectory, and suplementary information are stored in a subfolder
    molecule_path = os.path.join(root, name)

    # read out the number of atoms and residues from the topology file
    with open(os.path.join(molecule_path, "top.pdb"), 'r') as file:
        lines = file.readlines()
        lastline = lines[-3]
        n_atoms = int(lastline[4:11].strip())
        n_res = int(lastline[22:26].strip())
        print(f"Number of atoms: {n_atoms}, residues: {n_res}")

    # read additional information to the molecule
    with open(os.path.join(molecule_path, "info.yaml"), 'r') as file:
        info = yaml.load(file, yaml.FullLoader)

    system = Peptide(short=False, n_atoms=n_atoms, n_res=n_res, filepath=molecule_path)
    traj = md.load_hdf5(os.path.join(root, "traj.h5"))
    xyz = traj.xyz
    if not xyz.shape[-2] == n_atoms:
        raise ValueError(f"pdb file ({n_atoms} atoms) does not match the data ({xyz.shape[-2]} atoms).")
    xyz_as_tensor = Tensor(xyz).view(*xyz.shape[:-2], -1)
    temperature = info["temperature"]

    return system, xyz_as_tensor, temperature


class LoaderHParams(HParams):
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int | None = None

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        if hparams.train_split + hparams.val_split + hparams.test_split != 1:
            raise ValueError(
                f"The sum of train_split ({hparams.train_split}), val_split ({hparams.val_split}), and "
                f"test_split ({hparams.test_split}) must be 1.")
        return hparams


class Loader(ABC):
    hparams_type = LoaderHParams
    hparams: LoaderHParams

    def __init__(
            self,
            hparams: LoaderHParams | dict,
    ):
        if not isinstance(hparams, self.hparams_type):
            hparams = self.hparams_type(**hparams)
        self.hparams = hparams

    @abstractmethod
    def generate_datasets(self) -> datasets:
        raise NotImplementedError

    @property
    @abstractmethod
    def dims(self) -> int:
        raise NotImplementedError


class PeptideLoaderHParams(LoaderHParams):
    root: str
    name: str
    method: Choice("bgmol", "h5")

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        if not os.path.abspath(hparams.root):
            default_data_path = os.path.join(get_default_path(), "data")
            root = os.path.join(default_data_path, hparams.root)
            warnings.warn(f"root path '{hparams.root}' is not an absolute path. Default data path '{default_data_path}'"
                          f" is assumed.")
            hparams.root = root
        return hparams


class PeptideLoader(Loader):
    hparams_type = PeptideLoaderHParams
    hparams: PeptideLoaderHParams

    def __init__(
            self,
            hparams: PeptideLoaderHParams | dict,
    ):
        super().__init__(hparams)
        system, xyz_as_tensor, temperature = self.load()
        # need to reinitialize the energy model to set n_workers to 1 due to a bug that prevents multiple workers from
        # shutting down properly after the training is done https://github.com/noegroup/bgflow/issues/35
        system.reinitialize_energy_model(temperature=temperature, n_workers=1)
        self.system = system
        self.data = xyz_as_tensor
        self.temperature = temperature
        self.energy_function = system.energy_model.energy

    def generate_datasets(self):
        # shuffle the dataset indices around
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        # set the seed if given. We choose this method over torch.manual_seed to avoid changing the seed of the
        # global RNG
        rng = random.Random(self.hparams.seed)
        rng.shuffle(indices)

        # calculate the split sizes
        train_split = int(self.hparams.train_split * dataset_size)
        val_split = int(self.hparams.val_split * dataset_size)
        test_split = dataset_size - train_split - val_split
        # sanity test
        assert dataset_size * self.hparams.test_split - 2 <= test_split <= dataset_size * self.hparams.test_split + 2, \
            "something has gone wrong with the calculation of test_split."

        # split the dataset
        train_indices = indices[:train_split]
        val_indices = indices[train_split:train_split + val_split]
        test_indices = indices[train_split + val_split:]

        return (
            PeptideCCDataset(self.data[train_indices]),
            PeptideCCDataset(self.data[val_indices]),
            PeptideCCDataset(self.data[test_indices])
        )

    @property
    def dims(self) -> int:
        return self.data.shape[-1]

    def load(self):
        if self.hparams.method == "bgmol":
            return load_from_bgmol(self.hparams.root, self.hparams.name)
        elif self.hparams.method == "h5":
            return load_from_h5(self.hparams.root, self.hparams.name)
        else:
            raise ValueError(f"Method {self.hparams.method} not recognized.")


class ToyLoaderHParams(LoaderHParams):
    name: str
    samples: int
    kwargs: dict = {}

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        if "max_samples" in hparams.kwargs.keys():
            raise ValueError("max_samples should not be specified in kwargs. Instead specify the number of samples in "
                             "and the split of training, validation and test data.")
        return hparams


class ToyLoader(Loader):
    hparams_type = ToyLoaderHParams
    hparams: ToyLoaderHParams

    def generate_datasets(self):
        # calculate the split sizes
        train_split = int(self.hparams.train_split * self.hparams.samples)
        val_split = int(self.hparams.val_split * self.hparams.samples)
        test_split = self.hparams.samples - train_split - val_split
        # sanity test
        assert (self.hparams.samples * self.hparams.test_split - 2 <= test_split <=
                self.hparams.samples * self.hparams.test_split + 2), \
            "something has gone wrong with the calculation of test_split."

        ToyDataSet = getattr(lt.datasets.toy, self.hparams.name)
        return (
            ToyDataSet(max_samples=train_split, **self.hparams.kwargs),
            ToyDataSet(max_samples=val_split, **self.hparams.kwargs),
            ToyDataSet(max_samples=test_split, **self.hparams.kwargs)
        )

    @property
    def dims(self) -> int:
        if self.hparams.name == "MoonsDataset" or self.hparams.name == "CirclesDataset":
            return 2
        else:
            return self.hparams.kwargs["dimensions"]
