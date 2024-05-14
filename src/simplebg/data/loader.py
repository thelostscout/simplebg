import os
import random
import yaml

from torch import Tensor
from torch.utils.data import TensorDataset, Subset

import bgmol
from bgmol.systems.peptide import peptide
import mdtraj as md

from lightning_trainable.hparams import HParams, AttributeDict
from lightning_trainable.hparams.types import Choice

from .dataset import PeptideCCDataset


def load_from_bgmol(
        root: str,
        name: str,
):
    DataSetClass = getattr(bgmol.datasets, name)
    # extract filename from url and replace .tgz with .npy
    filename = os.path.splitext(DataSetClass.url)[0] + ".npy"
    path = os.path.join(root, filename)
    download = not os.path.exists(path)
    DataSet = DataSetClass(root=root, download=download, read=True)

    xyz_as_tensor = Tensor(DataSet.xyz).view(*DataSet.xyz.shape[:-2], -1)

    return DataSet.system, xyz_as_tensor, DataSet.temperature


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

    system = peptide(short=False, n_atoms=n_atoms, n_res=n_res, filepath=molecule_path)
    traj = md.load_hdf5(os.path.join(root, "traj.h5"))
    xyz = traj.xyz
    if not xyz.shape[-2] == n_atoms:
        raise ValueError(f"pdb file ({n_atoms} atoms) does not match the data ({xyz.shape[-2]} atoms).")
    xyz_as_tensor = Tensor(xyz).view(*xyz.shape[:-2], -1)
    temperature = info["temperature"]

    return system, xyz_as_tensor, temperature


class PeptideLoaderHParams(HParams):
    root: str
    name: str
    method: Choice("bgmol", "h5")


class PeptideLoader:
    hparams_type = PeptideLoaderHParams
    hparams: PeptideLoaderHParams

    def __init__(
            self,
            hparams: PeptideLoaderHParams | dict,
    ):
        if not isinstance(hparams, self.hparams_type):
            hparams = self.hparams_type(**hparams)
        self.hparams = hparams
        system, xyz, temperature = self.load()
        # need to reinitialize the energy model to set n_workers to 1 due to a bug that prevents multiple workers from
        # shutting down properly after the training is done https://github.com/noegroup/bgflow/issues/35
        system.reinitialize_energy_model(temperature=temperature, n_workers=1)
        self.system = system
        self.cartesian = PeptideCCDataset(xyz)
        self.temperature = temperature

    def load(self):
        if self.hparams.method == "bgmol":
            return load_from_bgmol(self.hparams.root, self.hparams.name)
        elif self.hparams.method == "h5":
            return load_from_h5(self.hparams.root, self.hparams.name)
        else:
            raise ValueError(f"Method {self.hparams.method} not recognized.")


class DataSplitHParams(HParams):
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int | None = None

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)
        if hparams.train_split + hparams.val_split + hparams.test_split != 1:
            raise ValueError(f"The sum of train_split ({hparams.train_split}), val_split ({hparams.val_split}), and "
                             f"test_split ({hparams.test_split}) must be 1.")
        return hparams


def split_dataset(
        dataset: TensorDataset,
        hparams: DataSplitHParams | dict = None,
):
    if not isinstance(hparams, DataSplitHParams):
        if hparams is None:
            hparams = DataSplitHParams()
        else:
            hparams = DataSplitHParams(**hparams)

    # shuffle the dataset indices around
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # set the seed if given. We choose this method over torch.manual_seed to avoid changing the seed of the global RNG
    rng = random.Random(hparams.seed)
    rng.shuffle(indices)

    # calculate the split sizes
    train_split = int(hparams.train_split * dataset_size)
    val_split = int(hparams.val_split * dataset_size)
    test_split = dataset_size - train_split - val_split
    # sanity test
    assert dataset_size * hparams.test_split - 2 <= test_split <= dataset_size * hparams.test_split + 2, \
        "something has gone wrong with the calculation of test_split."

    # split the dataset
    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]

    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)

    return train_data, val_data, test_data
