import math

import mdtraj
import torch

from FrEIA.utils import force_to


# TODO: use a premade backbone searcher
def align_backbone(coordinates, system):
    df = system.mdtraj_topology.to_dataframe()[0]
    atom_idx = (df['element'] != "H").astype(int).to_numpy().nonzero()[0]
    traj = mdtraj.Trajectory(coordinates, system.mdtraj_topology)
    traj = traj.superpose(traj, frame=0, atom_indices=atom_idx, parallel=True)
    return traj.xyz


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


class Alignment:
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
        return torch.mean((x - traj.xyz) ** 2, dim=1)
