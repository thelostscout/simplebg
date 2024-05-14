"""
This module contains various internal coordinate (IC) transforms.
Transformations should map a data.dataset to another.
"""

import bgmol
import bgflow
from ..data.dataset import *


class CartesianToInternalTransform:
    def __init__(self, system, normalize_angles=True):
        zfactory = bgmol.zmatrix.ZMatrixFactory(system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        self.transform = bgflow.GlobalInternalCoordinateTransformation(zmatrix, normalize_angles=normalize_angles)

    def forward(self, dataset: PeptideCCDataset):
        bonds, angles, torsions, origin, rotation, log_det_j = self.transform.forward(dataset.tensors[0])
        return PeptideICDataset(bonds, angles, torsions, origin.squeeze(), rotation), log_det_j

    def inverse(self, dataset: PeptideICDataset):
        coordinates, log_det_j = self.transform.inverse(
            bonds=dataset.bonds,
            angles=dataset.angles,
            torsions=dataset.torsions,
            x0=dataset.origin.unsqueeze(0),
            R=dataset.rotation,
        )
        return PeptideCCDataset(coordinates), log_det_j
