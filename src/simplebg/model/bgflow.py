import numpy as np
import torch
import scipy

import bgflow
from bgflow.factory.tensor_info import TORSIONS, ANGLES, BONDS
import bgmol
import lightning_trainable as lt
from lightning_trainable.hparams import HParams

from . import core
from .. import network, latent
from ..data import loaders


class NetworkHParams(HParams):
    nontrivial_torsions_depth: int
    nontrivial_torsions_kwargs: dict = {"hidden": (512, 512)}
    trivial_torsions_depth: int
    trivial_torsions_kwargs: dict = {"hidden": (512, 512)}
    angles_depth: int
    angles_kwargs: dict = {"hidden": (512, 512)}
    bonds_depth: int
    bonds_kwargs: dict = {"hidden": (512, 512)}


class FlowHParams(core.BaseHParams):
    model_class = "FlowModel"
    shapiro_threshold: float
    network_hparams: NetworkHParams

    @classmethod
    def validate_parameters(cls, hparams):
        return super(lt.TrainableHParams, cls).validate_parameters(hparams)


class FlowModel(core.BaseModel):
    hparams_type = FlowHParams
    hparams: FlowHParams
    loader_class = loaders.PeptideLoader

    def __init__(self, hparams):
        super().__init__(hparams)
        self.generator, self.nn, self.q = self.build()

    def build(self):
        zfactory = bgmol.ZMatrixFactory(self.peptide.system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        coordinate_transform = bgflow.GlobalInternalCoordinateTransformation(zmatrix)
        torsions_values = np.array(coordinate_transform.forward(self.peptide.data)[2].detach().numpy())
        n_torsions = torsions_values.shape[-1]
        is_gaussian = np.full((n_torsions,), False, dtype=bool)
        is_tobe_shifted = np.full((n_torsions,), False, dtype=bool)
        for i, torsion_values in enumerate(torsions_values.transpose()):
            pvalue = scipy.stats.shapiro(torsion_values)[1]
            if pvalue > self.hparams.shapiro_threshold:
                is_gaussian[i] = True
            else:
                pvalue = scipy.stats.shapiro((torsion_values + .5) % 1)[1]
                if pvalue > self.hparams.shapiro_threshold:
                    is_tobe_shifted[i] = True
        is_gaussian = np.insert(is_gaussian, 0, [False, False, False])
        is_tobe_shifted = np.insert(is_tobe_shifted, 0, [False, False, False])
        is_trivial = is_gaussian + is_tobe_shifted
        n_trivialt = is_trivial.sum()
        zmatrix = np.vstack((zmatrix[~is_trivial], zmatrix[is_gaussian], zmatrix[is_tobe_shifted]))
        coordinate_transform = bgflow.GlobalInternalCoordinateTransformation(zmatrix)
        constrained_indices, constrained_lengths = bgmol.bond_constraints(
            self.peptide.system.system,
            coordinate_transform
        )
        shape_info = bgflow.ShapeDictionary.from_coordinate_transform(
            coordinate_transform,
            n_constraints=self.peptide.system.system.getNumConstraints()
        )
        torsions_shift = torch.zeros(torsions_values.shape[-1])
        torsions_shift[-is_tobe_shifted.sum():] = 0.5
        prepareTorsions = network.bgflow.CircularShiftFlow(torsions_shift)
        torsions_values = coordinate_transform.forward(self.peptide.data)[2]
        trivial_torsions_values = prepareTorsions(torsions_values)[0][:, -n_trivialt:]
        trivial_torsions_marginal = bgflow.TruncatedNormalDistribution(
            mu=trivial_torsions_values.mean(axis=0),
            sigma=trivial_torsions_values.std(axis=0),
            lower_bound=torch.tensor(0),
            upper_bound=torch.tensor(1),
        )
        NONTRIVIAL_TORSIONS = bgflow.TensorInfo(name='NONTRIVIAL_TORSIONS', is_circular=True)
        TRIVIAL_TORSIONS = bgflow.TensorInfo(name='TRIVIAL_TORSIONS', is_circular=True)

        shape_info.split(
            TORSIONS,
            into=(NONTRIVIAL_TORSIONS, TRIVIAL_TORSIONS),
            sizes=(n_torsions - n_trivialt, n_trivialt)
        )

        builder = bgflow.BoltzmannGeneratorBuilder(shape_info, self.peptide.system.energy_model)

        # first, work on torsions, as they are the hardest part
        t1, t2 = builder.add_split(
            NONTRIVIAL_TORSIONS,
            into=["T1", "T2"],
            sizes_or_indices=[shape_info[NONTRIVIAL_TORSIONS][0] // 2,
                              shape_info[NONTRIVIAL_TORSIONS][0] - shape_info[NONTRIVIAL_TORSIONS][0] // 2],
        )
        for i in range(self.hparams.network_hparams.nontrivial_torsions_depth):
            builder.add_condition(t1, on=t2, **self.hparams.network_hparams.nontrivial_torsions_kwargs)
            builder.add_condition(t2, on=t1, **self.hparams.network_hparams.nontrivial_torsions_kwargs)

        builder.add_merge(
            (t1, t2),
            to=NONTRIVIAL_TORSIONS,
            sizes_or_indices=[shape_info[NONTRIVIAL_TORSIONS][0] // 2,
                              shape_info[NONTRIVIAL_TORSIONS][0] - shape_info[NONTRIVIAL_TORSIONS][0] // 2],
        )
        t1, t2 = builder.add_split(
            TRIVIAL_TORSIONS,
            into=["T1", "T2"],
            sizes_or_indices=[shape_info[TRIVIAL_TORSIONS][0] // 2,
                              shape_info[TRIVIAL_TORSIONS][0] - shape_info[TRIVIAL_TORSIONS][0] // 2],
        )
        for i in range(self.hparams.network_hparams.trivial_torsions_depth):
            builder.add_condition(t1, on=(t2, NONTRIVIAL_TORSIONS),
                                  **self.hparams.network_hparams.trivial_torsions_kwargs,
                                  )
            builder.add_condition(t2, on=(t1, NONTRIVIAL_TORSIONS),
                                  **self.hparams.network_hparams.trivial_torsions_kwargs,
                                  )
        builder.add_merge(
            (t1, t2),
            to=TRIVIAL_TORSIONS,
            sizes_or_indices=[shape_info[TRIVIAL_TORSIONS][0] // 2,
                              shape_info[TRIVIAL_TORSIONS][0] - shape_info[TRIVIAL_TORSIONS][0] // 2],
        )

        # then, do angles and bonds
        a1, a2 = builder.add_split(
            ANGLES,
            into=["A1", "A2"],
            sizes_or_indices=[shape_info[ANGLES][0] // 2, shape_info[ANGLES][0] - shape_info[ANGLES][0] // 2],
        )
        for i in range(self.hparams.network_hparams.angles_depth):
            builder.add_condition(a1, on=[a2, NONTRIVIAL_TORSIONS, TRIVIAL_TORSIONS],
                                  **self.hparams.network_hparams.angles_kwargs,
                                  )
            builder.add_condition(a2, on=[a1, NONTRIVIAL_TORSIONS, TRIVIAL_TORSIONS],
                                  **self.hparams.network_hparams.angles_kwargs,
                                  )
        builder.add_merge(
            (a1, a2),
            to=ANGLES,
            sizes_or_indices=[shape_info[ANGLES][0] // 2, shape_info[ANGLES][0] - shape_info[ANGLES][0] // 2],
        )

        b1, b2 = builder.add_split(
            BONDS,
            into=["B1", "B2"],
            sizes_or_indices=[shape_info[BONDS][0] // 2, shape_info[BONDS][0] - shape_info[BONDS][0] // 2],
        )
        for i in range(self.hparams.network_hparams.bonds_depth):
            builder.add_condition(b1, on=[b2, ANGLES, NONTRIVIAL_TORSIONS, TRIVIAL_TORSIONS],
                                  **self.hparams.network_hparams.bonds_kwargs,
                                  )
            builder.add_condition(b2, on=[b1, ANGLES, NONTRIVIAL_TORSIONS, TRIVIAL_TORSIONS],
                                  **self.hparams.network_hparams.bonds_kwargs,
                                  )
        builder.add_merge(
            (b1, b2),
            to=BONDS,
            sizes_or_indices=[shape_info[BONDS][0] // 2, shape_info[BONDS][0] - shape_info[BONDS][0] // 2],
        )
        cdfs = bgflow.InternalCoordinateMarginals(  # Annahmen der cumulative density function
            builder.current_dims,
            builder.ctx,
            bonds=BONDS,
            angles=ANGLES,
            torsions=None,  # TORSIONS,
            fixed=None,
            bond_mu=0.2,
            bond_sigma=2.0,
            bond_upper=1.0,
            bond_lower=0.03,
            # angle_mu=0.7,
            angle_lower=0.1,
            angle_upper=0.9
        )
        cdfs.inform_with_data(
            self.peptide.data[:1_000], coordinate_transform,
            constrained_bond_indices=bgmol.bond_constraints(self.peptide.system.system, coordinate_transform)[0]
        )

        cdfs[TRIVIAL_TORSIONS] = trivial_torsions_marginal
        icdf_maps = builder.add_map_to_ic_domains(cdfs, return_layers=True)

        builder.add_merge([NONTRIVIAL_TORSIONS, TRIVIAL_TORSIONS], to=TORSIONS)
        fmod_layer = network.bgflow.CircularShiftFlow(torsions_shift)
        builder.add_layer(fmod_layer, what=(TORSIONS,))

        builder.add_merge_constraints(*bgmol.bond_constraints(self.peptide.system.system, coordinate_transform))
        builder.add_map_to_cartesian(coordinate_transform)
        generator = builder.build_generator()
        dims_out = tuple(int(size[0]) for size in generator.prior.event_shapes)
        nn = network.bgflow.NetworkWrapper(generator._flow, dims_in=self.peptide.data.shape[-1], dims_out=dims_out)
        q = latent.PriorWrapper(generator.prior)
        return generator, nn, q

    @property
    def peptide(self):
        return self.loader
