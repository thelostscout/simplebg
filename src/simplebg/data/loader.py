import os.path

import bgmol
import mdtraj as md
import yaml
from bgmol.systems.peptide import peptide
from lightning_trainable.hparams import HParams
from lightning_trainable.hparams.types import Choice


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

    return DataSet.system, DataSet.xyz, DataSet.temperature


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
    assert xyz.shape[-2] == n_atoms, f"pdb file ({n_atoms} atoms) does not match the data ({xyz.shape[-2]} atoms)."
    temperature = info["temperature"]

    return system, xyz, temperature


class PeptideLoaderHParams(HParams):
    root: str
    name: str
    method: Choice("bgmol", "h5")


class PeptideLoader:
    hparams: PeptideLoaderHParams

    def __init__(self, hparams):
        self.hparams = hparams

    def load(self):
        if self.hparams.method == "bgmol":
            return load_from_bgmol(self.hparams.root, self.hparams.name)
        elif self.hparams.method == "h5":
            return load_from_h5(self.hparams.root, self.hparams.name)
        else:
            raise ValueError(f"Method {self.hparams.method} not recognized.")
