import math
from typing import Iterable, Tuple

import mdtraj
import torch
import bgmol
import bgflow

from FrEIA.utils import force_to
from torch import Tensor
import FrEIA.modules as Fm
import bgflow
import os


# TODO: use a premade backbone searcher
def align_backbone(coordinates, system):
    df = system.mdtraj_topology.to_dataframe()[0]
    atom_idx = (df['element'] != "H").astype(int).to_numpy().nonzero()[0]
    traj = mdtraj.Trajectory(coordinates, system.mdtraj_topology)
    traj = traj.superpose(traj, frame=0, atom_indices=atom_idx, parallel=True)
    return traj.xyz


def load_data(data_path):
    if "alanine_dipeptide" in data_path:
        is_data_here = os.path.exists(data_path + "/Ala2TSF300.npy")
        ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=data_path)
        system = ala_data.system
        coordinates = ala_data.coordinates
    else:
        with open(data_path.rstrip("/") + "/top.pdb", 'r') as file:
            lines = file.readlines()
            lastline = lines[-3]
            n_atoms = int(lastline[4:11].strip())
            n_res = int(lastline[22:26].strip())
            system = bgmol.systems.peptide(short=False, n_res=n_res, n_atoms=n_atoms, filepath=data_path)
            traj = mdtraj.load_hdf5(data_path.rstrip("/") + "/traj.h5")
            coordinates = traj.xyz
    system.reinitialize_energy_model(temperature=300., n_workers=1)
    return coordinates, system


def load_model_kwargs(ModelClass, train_data, val_data, system):
    kwargs = dict(train_data=train_data, val_data=val_data)
    if ModelClass.needs_energy_function:
        kwargs['energy_function'] = system.energy_model.energy
        if ModelClass.needs_alignment:
            kwargs['alignment_penalty'] = AlignmentIC(system).penalty
    elif ModelClass.needs_system:
        kwargs['system'] = system
    return kwargs


class SingleTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, tensor):
        super().__init__(tensor)

    def __getitem__(self, index):
        return self.tensors[0][index]

    def get_tensor(self):
        return self.tensors[0]

    def to(self, *args, **kwargs):
        force_to(self.q, *args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        # Delegate to .to(...) as it moves distributions, too
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        return self.to(device)

    def cpu(self):
        # Delegate to .to(...) as it moves distributions, too
        return self.to("cpu")


class CoordinateDataset(SingleTensorDataset):
    def __init__(self, coordinates, system, mode, val_split=.1, test_split=.2, seed=42):

        self.mode = mode

        N = coordinates.shape[0]
        coordinates = align_backbone(coordinates, system)
        coordinate_tensor = torch.Tensor(coordinates.reshape(N, -1))

        self.reference_molecule = coordinates[0]

        val_len = int(math.floor(N * val_split))
        test_len = int(math.floor(N * test_split))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(N, generator=generator)

        if mode == 'test':
            idx = indices[:test_len]
        elif mode == 'val':
            idx = indices[test_len: test_len + val_len]
        elif mode == 'train':
            idx = indices[test_len + val_len:]
        elif mode == 'no_split':
            idx = indices
        else:
            raise ValueError(f"{mode} is not a valid mode. Choose from ['train', 'test', 'val', 'no_split']")

        super().__init__(coordinate_tensor[idx])


def dataset_setter(coordinates, system, val_split=.1, test_split=.2, seed=42):
    modes = ['train', 'val', 'test']
    out = [CoordinateDataset(coordinates, system, mode, val_split, test_split, seed) for mode in modes]
    # print("mode order: " + s[:-2])
    return out


class AlignmentRMS:
    def __init__(self, system, reference_molecule):
        self.reference_molecule = mdtraj.Trajectory(reference_molecule, system.mdtraj_topology)
        df = system.mdtraj_topology.to_dataframe()[0]
        self.atom_idx = (df['element'] != "H").astype(int).to_numpy().nonzero()[0]

    def penalty(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.cpu()
            x = x.reshape(*x.shape[:-1], *self.reference_molecule.xyz.shape[-2:])
            traj = mdtraj.Trajectory(x, self.reference_molecule.topology)
            traj = traj.superpose(self.reference_molecule, frame=0, atom_indices=self.atom_idx, parallel=True)
        return torch.mean((x - traj.xyz).reshape(*x.shape[:-2], -1) ** 2, dim=-1)


class AlignmentIC:
    def __init__(self, system):
        zfactory = bgmol.zmatrix.ZMatrixFactory(system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        self.ic_layer = bgflow.GlobalInternalCoordinateTransformation(zmatrix)

    def penalty(self, x: torch.Tensor):
        _0, _1, _2, loc, rot, _5 = self.ic_layer(x)
        loc = loc.squeeze()
        return torch.sum(loc ** 2, dim=-1), torch.sum(rot ** 2, dim=-1)


class ICTransform(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, system=None):
        super().__init__(dims_in, dims_c)
        zfactory = bgmol.zmatrix.ZMatrixFactory(system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        self.bg_layer = bgflow.GlobalInternalCoordinateTransformation(zmatrix)

    def output_dims(self, input_dims):
        return input_dims

    def forward(
            self, x_or_z: Iterable[Tensor],
            c: Iterable[Tensor] = None,
            rev: bool = False,
            jac: bool = True
    ) -> Tuple[Tuple[Tensor], Tensor]:
        x_or_z = x_or_z[0]
        if not rev:
            bonds, angles, torsions, loc, rot, log_jac_det = self.bg_layer._forward(x_or_z)
            out = torch.cat([bonds, angles, torsions, loc.squeeze(1), rot], dim=1)
        else:
            bonds = x_or_z[:, :self.bg_layer.dim_bonds]
            angles = x_or_z[:, self.bg_layer.dim_bonds:self.bg_layer.dim_bonds + self.bg_layer.dim_angles]
            torsions = x_or_z[:,
                              self.bg_layer.dim_bonds + self.bg_layer.dim_angles:self.bg_layer.dim_bonds + self.bg_layer.dim_angles + self.bg_layer.dim_torsions]
            loc = x_or_z[-6:-3].unsqueeze(1)
            rot = x_or_z[-3:]
            out, log_jac_det = self.bg_layer._inverse(bonds, angles, torsions, loc, rot)
        return (out,), log_jac_det.squeeze(1)
