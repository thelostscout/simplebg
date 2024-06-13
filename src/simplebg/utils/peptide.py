import os
import pickle

import numpy as np
from bgmol.util.importing import import_openmm
from bgmol.systems.base import OpenMMSystem
from openmm.app import *
import mdtraj as md
import numpy
_, unit, app = import_openmm()

__all__ = ["peptide"]


class peptide(OpenMMSystem):
    """1b5j trimer with #TODO force field in implicit solvent.

    Attributes
    ----------
    constraints : app.internal.singleton.Singleton
        Constraint types
    hydrogen_mass : unit.Quantity or None
        If None, don't repartition hydrogen mass. Else, assign the specified mass to hydrogen atoms.
    implicit_solvent : app.internal.singleton.Singleton or None
        Implicit solvent model to be used.
    root : str
        where are the files?

    Notes
    -----
    TODO: positions, maybe?, TICA
    """

    def __init__(
            self,
            n_atoms: int,
            n_res: int,
            filepath: str,
            short=True,
            explicit=False,
            constraints=app.HBonds,
            hydrogen_mass=4.0 * unit.amu

    ):
        super().__init__()

        if short:
            self.root = filepath
            self._positions = numpy.zeros((n_atoms, n_res))
        else:
            self.root = filepath
            self._positions = numpy.zeros((n_atoms, n_res))
        # if ga:
        # self.root = os.path.join(self.root,"GaMD")
        # else:
        # self.root = os.path.join(self.root,"cMD")
        if explicit:
            self.root = os.path.join(self.root, "Explicit_Solvent")
        else:
            self.root = os.path.join(self.root)
            self.implicit_solvent = self.system_parameter(
                "implicit_solvent", app.GBn2, default=app.GBn2
            )

        # if alpha_helix:
        # self.root = os.path.join(self.root,"alpha_helix")
        # else:
        # self.root = os.path.join(self.root,"extended")

        self.files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]

        self.constraints = self.system_parameter(
            "constraints", constraints, default=app.HBonds
        )
        self.hydrogen_mass = self.system_parameter(
            "hydrogen_mass", hydrogen_mass, default=4.0 * unit.amu
        )

        for sourcefile in self.files:
            os.path.isfile(os.path.join(self.root, sourcefile))
            assert os.path.isfile(os.path.join(self.root, sourcefile))

        top_file = os.path.join(self.root, "top.pdb")
        modeller = PDBFile(top_file)

        forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')

        self._system = forcefield.createSystem(modeller.topology,
                                               hydrogenMass=hydrogen_mass,
                                               constraints=constraints)
        #implicitSolvent = self.implicit_solvent)

        self._topology = modeller.topology
        self._mdtraj_topology = md.Topology().from_openmm(self._topology)

        #TODO: TICA
        # Load the CHARMM files

    #      params = app.CharmmParameterSet(
    #          os.path.join(root, "top_all22star_prot.rtf"),
    #          os.path.join(root, "top_water_ions.rtf"),
    #          os.path.join(root, "parameters_ak_dihefix.prm")
    #      )

    #      # create system
    #      with fixed_atom_names(TYR=["HT1", "HT2", "HT3"]):
    #          psf = app.CharmmPsfFile(os.path.join(root, "structure.psf"))
    #          crds = app.PDBFile(os.path.join(root, "structure.pdb"))
    #      self._system = psf.createSystem(
    #          params,
    #          nonbondedMethod=app.NoCutoff,
    #          constraints=constraints,
    #          hydrogenMass=hydrogen_mass,
    #          implicitSolvent=implicit_solvent
    #      )
    #      self._positions = crds.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    #      self._topology = psf.topology

    #        self._tica_mean, self._tica_eig = self._read_tica(root)

    def _read_tica(self, filename="tica.npz"):
        with open(filename, "rb") as f:
            self.tica_model = pickle.load(f)
        # npz = np.load(filename,allow_pickle=True)
    # self._tica_mean, self._tica_eig = npz["tica_mean"], npz["tica_eigenvectors"]
    # return npz["tica_mean"], npz["tica_eigenvectors"]


#
#  FILES = {
#      "parameters_ak_dihefix.prm": "f712c2392fdf892e43341fed64305ba8",
#      "structure.pdb": "be19629a75e0ee4e1cc3c72a9ebc63c6",
#      "structure.psf": "944b26edb992c7dbdaa441675b9e42c5",
#      "top_all22star_prot.rtf": "d046c9a998369be142a6470fd5bb3de1",
#      "top_water_ions.rtf": "ade085f88e869de304c814bf2d0e57fe",
#      "chignolin_tica.npz": "9623ea5b73f48b6952db666d586a27d6"
#  }

# def to_tics(self, xs, eigs_kept=None):
#   c_alpha = self.mdtraj_topology.select("backbone and (name C or name N)")
#   xs = xs.reshape(xs.shape[0], -1, 3)
#   xs = xs[:, c_alpha, :]
#   if eigs_kept is None:
#       eigs_kept = self._tica_eig.shape[-1]
#   dists = all_distances(xs)
#   return (dists - self._tica_mean) @ self._tica_eig[:, :eigs_kept]


def all_distances(xs):
    if isinstance(xs, np.ndarray):
        mask = np.triu(np.ones([xs.shape[-2], xs.shape[-2]]), k=1).astype(bool)
        xs2 = np.square(xs).sum(axis=-1)
        ds2 = xs2[..., None] + xs2[..., None, :] - 2 * np.einsum("nid, njd -> nij", xs, xs)
        ds2 = ds2[:, mask].reshape(xs.shape[0], -1)
        ds = np.sqrt(ds2)
    else:
        import torch
        assert isinstance(xs, torch.Tensor)
        mask = torch.triu(torch.ones([xs.shape[-2], xs.shape[-2]]), diagonal=1).bool()
        xs2 = xs.pow(2).sum(dim=-1)
        ds2 = xs2[..., None] + xs2[..., None, :] - 2 * torch.einsum("nid, njd -> nij", xs, xs)
        ds2 = ds2[:, mask].view(xs.shape[0], -1)
        ds = ds2.sqrt()
    return ds
