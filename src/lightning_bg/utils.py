import math

import mdtraj
import torch

# This would be obsolete
def remove_H_atoms(coordinates, system):
    df = system.mdtraj_topology.to_dataframe()[0]
    atom_idx = (df['element'] != "H").astype(int).to_numpy().nonzero()
    return coordinates[:, atom_idx].squeeze()


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


class CoordinateDataset(SingleTensorDataset):
    def __init__(self, coordinates, system, mode, val_split=.1, test_split=.2, seed=42):

        self.mode = mode

        N = coordinates.shape[0]
        coordinates = align_backbone(coordinates, system)
        coordinate_tensor = torch.Tensor(coordinates.reshape(N, -1))

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


def tensor_analysis(tensor):
    print(f"size: {tensor.shape}, {tensor.min():3f} to {tensor.max():3f}, mean: {tensor.mean():3f} +/- {tensor.var().sqrt():3f}")
