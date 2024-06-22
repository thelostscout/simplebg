import torch
from torch import nn

import bgmol
import bgflow
from lightning_trainable.hparams import AttributeDict


class Transform(nn.Module):
    def forward(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    def reverse(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


# TODO: this might cause problems, need to monitor

class InverseTransform(Transform):
    def __init__(self, reference_transform: Transform):
        super().__init__()
        self._reference = reference_transform

    def forward(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self._reference.reverse(input)

    def reverse(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self._reference.forward(input)


class IdentityTransform(Transform):
    def forward(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return input, torch.zeros_like(input)

    def reverse(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return input, torch.zeros_like(input)


class GlobalInternalCoordinateTransformation(Transform):
    def __init__(
            self,
            system: bgmol.systems.OpenMMSystem,
            normalize_angles: bool = True,
    ):
        super().__init__()
        zfactory = bgmol.zmatrix.ZMatrixFactory(system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        self._bg_layer = bgflow.GlobalInternalCoordinateTransformation(
            zmatrix,
            normalize_angles=normalize_angles,
            raise_warnings=True,
        )

    @property
    def z_matrix(self):
        return self._bg_layer.z_matrix

    @property
    def fixed_atoms(self):
        return self._bg_layer.fixed_atoms

    @property
    def dim_bonds(self):
        return self._bg_layer.dim_bonds

    @property
    def dim_angles(self):
        return self._bg_layer.dim_angles

    @property
    def dim_torsions(self):
        return self._bg_layer.dim_torsions

    @property
    def dim_fixed(self):
        return self._bg_layer.dim_fixed

    @property
    def bond_indices(self):
        return self._bg_layer.bond_indices

    @property
    def angle_indices(self):
        return self._bg_layer.angle_indices

    @property
    def torsion_indices(self):
        return self._bg_layer.torsion_indices

    @property
    def normalize_angles(self):
        return self._bg_layer.normalize_angles

    def forward(self, input: torch.Tensor, rev=True) -> (torch.Tensor, torch.Tensor):
        bonds, angles, torsions, x0, R, log_det_j = self._bg_layer._forward(input)
        return torch.cat([bonds, angles, torsions, x0.squeeze(-2), R], dim=-1), log_det_j.squeeze(-1)

    def reverse(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        split_dims = [self.dim_bonds, self.dim_angles, self.dim_torsions, 3, 3]
        bonds, angles, torsions, x0, R = torch.split(input, split_dims, dim=-1)
        output, log_det_j = self._bg_layer._inverse(bonds, angles, torsions, x0.unsqueeze(-2), R)
        return output, log_det_j.squeeze(-1)


classes = dict(
    ic=GlobalInternalCoordinateTransformation,
    identity=IdentityTransform,
)


def constructor(transform_name: str, **kwargs):
    TransformClass = classes[transform_name]
    return TransformClass(**kwargs)
