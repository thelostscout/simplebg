import bgmol
import ipywidgets as ipw
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj
import nglview
import numpy as np
import torch
from bgmol.systems.ala2 import compute_phi_psi


class ShowTraj(ipw.VBox):
    """Combines the nglview show_traj widget with a text widget that shows the energy of the current frame."""

    def __init__(self, traj: mdtraj.Trajectory, energies: torch.Tensor, **kwargs):
        """
        Initialize the widget with any trajectory and a list of corresponding energies.
        :param traj: Any mdtraj trajectory.
        :param energies: A list of energies corresponding to the frames in the trajectory.
        :param kwargs: Any kwargs for the ipywidgets.VBox.
        """
        self.energies = energies

        self.wnumber = ipw.BoundedFloatText(
            value=np.around(self.energies[0].item(), decimals=2),
            description='Energy:',
            disabled=False,
            min=-1e20,
            max=1e50
        )
        self.wtraj = nglview.show_mdtraj(traj)
        self.wtraj._iplayer._trait_values["children"][1].observe(self.on_value_change, names='value')

        super().__init__(children=[self.wtraj, self.wnumber], **kwargs)

    def from_system(self, samples: torch.Tensor, system: bgmol.systems.OpenMMSystem, **kwargs):
        """
        Initialize the widget with samples from data space and the corresponding system.
        :param samples: Samples of the shape (-1, dim(data space)).
        :param system: The system that describes the molecule.
        :param kwargs: Any kwargs for the ipywidgets.VBox.
        """
        traj = self.create_traj(samples, system)
        system.reinitialize_energy_model(temperature=300., n_workers=1)
        energies = system.energy_model.energy(samples)
        self.__init__(traj, energies, **kwargs)

    @staticmethod
    def create_traj(samples, system):
        """Reshapes samples of flat dimension (dim(data space)) and turns it into a trajectory."""
        samples_flat = samples.reshape(len(samples), -1, 3).cpu().detach().numpy()
        return mdtraj.Trajectory(samples_flat, system.mdtraj_topology)

    def on_value_change(self, change):
        # update the energy text widget when the frame changes
        self.wnumber.value = np.around(self.energies[change['new']].item(), decimals=2)


def sample_energies(model, n_samples=1000):
    """Sample energies from a model."""
    if not hasattr(model, "peptide"):
        raise ValueError("Model does not have an energy method.")
    samples = model.sample((n_samples,))
    energies = model.peptide.system.energy_model.energy(samples)
    return energies.squeeze(1)
