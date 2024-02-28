import bgmol
import ipywidgets as ipw
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj
import nglview
import numpy as np
import torch
from bgmol.systems.ala2 import compute_phi_psi

from simplebg import models
from simplebg.utils import dataset_setter
from .utils import load_data, load_model_kwargs, SingleTensorDataset
import FrEIA.modules as Fm


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


class Evaluator:
    """
    A utility class that includes many methods within this module from a single initialization with a model and data.
    """

    def __init__(self, model: models.BaseTrainable, system: bgmol.systems.OpenMMSystem):
        self.inn = model.inn
        self.val_data = model.val_data
        self.system = system
        self.q = model.q
        self._all_funcs = [self.energy_plot, self.ramachandran_plot]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, data_path, model_class):
        """
        Load a model from a checkpoint.
        :param checkpoint_path: The path to the checkpoint.
        :param data_path: The path to the molecule data.
        :param model_class: The type of model.
        :return: An instance of this class.
        """
        # TODO: this should be reworked with a proper model loader that handles data loading etc. on its own.
        # load data with the data loader
        coordinates, system = load_data(data_path)
        train_split = .7
        train_data, val_data, test_data = dataset_setter(coordinates, system, val_split=(.8 - train_split),
                                                         test_split=.2, seed=42)
        # load the model class
        ModelClass = getattr(models, model_class)
        hparams = torch.load(checkpoint_path.rstrip("/") + "/checkpoints/last.ckpt")['hyper_parameters']
        # TODO: this is quite ugly and should be handled by the model class itself in the future
        model_kwargs = load_model_kwargs(ModelClass, train_data, val_data, system)
        model = ModelClass.load_from_checkpoint(checkpoint_path.rstrip("/") + "/checkpoints/last.ckpt",
                                                hparams=hparams,
                                                **model_kwargs
                                                )
        return cls(model, system)

    def energy_plot(self, rg=None, ax=None, **figkwargs):
        """Plot the energy distribution of generated data compared to the validation data."""
        return energy_plot(self.val_data, self.system.energy_model.energy, self.inn, self.q, rg, ax=ax, **figkwargs)

    def ramachandran_plot(self, axs=None, **figkwargs):
        """Plot a Ramachandran plot of generated data compared to the validation data."""
        return ramachandran_plot(self.val_data, self.system.mdtraj_topology, self.inn, self.q, axs=axs,
                                 **figkwargs)

    def plot_all(self, **figkwargs):
        outputs = []
        for f in self._all_funcs:
            # noinspection PyArgumentList
            out = f(**figkwargs)
            outputs.append(out)
        return outputs

    def plot_energy_and_ramachandran(self, save_loc, rg, **figkwargs):
        if 'figsize' not in figkwargs:
            figkwargs['figsize'] = (16, 5)
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, **figkwargs)
        self.energy_plot(rg=rg, ax=ax1)
        self.ramachandran_plot(axs=[ax2, ax3])
        fig.savefig(save_loc, bbox_inches='tight')
        pass

    def plot_energy(self, save_loc, rg, **figkwargs):
        fig, ax1 = plt.subplots(1, 1, **figkwargs)
        self.energy_plot(rg=rg, ax=ax1)
        fig.savefig(save_loc, bbox_inches='tight')
        pass


def energy_plot_simple(val_energies, sample_energies, rg, ax=None, **figkwargs):
    """
    Plot the energy distribution of generated data compared to the validation data.
    :param val_energies: Energies of the validation data.
    :param sample_energies: Energies of the generated data.
    :param rg: The range of the histogram.
    :param ax: The axis to plot on. If ``None``, a new figure is created.
    :param figkwargs: kwargs for a new figure if no axis is provided.
    :return: figure and axis.
    """
    # create a new figure if no axis is provided
    if ax is None:
        fig = plt.figure(**figkwargs)
        ax = plt.gca()
    else:
        fig = ax.figure
    # create two histograms that plot the energy distribution of the validation and generated data.
    ax.hist(val_energies.cpu().detach().numpy(), bins=200, histtype='step', range=rg, label=r"$p^*$")
    ax.hist(sample_energies.cpu().detach().numpy(), bins=200, histtype='step', range=rg, label=r"$\hat p$")
    # some plot styling
    ax.set_yscale('log')
    ax.set_ylabel("# samples")
    ax.set_xlabel(r"Energy [$k_B T$]")
    ax.legend()
    return fig, ax


def energy_plot(
        val_dataset: SingleTensorDataset,
        energy_function,
        INN: Fm.InvertibleModule,
        latent_target_distribution,
        rg,
        ax=None,
        **figkwargs
):
    """
    Plot the energy distribution of generated data compared to the validation data.
    :param val_dataset: The dataset that serves as comparison.
    :param energy_function: A function that calculates the energy of the data.
    :param INN: The invertible neural network that transforms the data.
    :param latent_target_distribution: The distribution q* in latent space.
    :param rg: The range of the histogram.
    :param ax: The axis to plot on. If ``None``, a new figure is created.
    :param figkwargs: kwargs for a new figure if no axis is provided.
    :return: The modified figure and axis.
    """
    # don't want to mess with the gradients while generating new samples --> no grad
    with torch.no_grad():
        # need same size datasets for proper comparison
        n_samples = len(val_dataset)
        z = latent_target_distribution.sample((n_samples,))
        # it would be nice if this could be done network agnostic
        x = INN(z, rev=True)[0]
        energies = energy_function(x)
        val_tensor = val_dataset.get_tensor()
        val_energies = energy_function(val_tensor)
    fig, ax = energy_plot_simple(val_energies, energies, rg, ax, **figkwargs)
    return fig, ax


# TODO: add plots for marginals of bonds etc. or different force term contributions

def plot_phi_psi(ax, phi_psi, bins=100):
    """
    Plot a 2d histogram of phi and psi angles.
    :param ax: The axis to plot on.
    :param phi_psi: A tuple of two arrays with phi and psi angles.
    :param bins: The ``bins`` argument for ``numpy.histogram2d``.
    """
    # some plot styling
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    cmap = mpl.cm.get_cmap('plasma').copy()
    cmap.set_bad(cmap(0))
    # plot one histrogram with the phi and psi angles
    ax.hist2d(
        phi_psi[0],
        phi_psi[1],
        bins=bins,
        density=True,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        cmap=cmap,
        norm=mpl.colors.LogNorm()
    )
    pass


def ramachandran_plot_simple(val_traj, sample_traj, axs=None, **figkwargs):
    """
    Plot a Ramachandran plot of generated data compared to the validation data.
    :param val_traj: Trajectory of the validation data.
    :param sample_traj: Trajectory of the generated data.
    :param axs: The axes to plot on. If ``None``, a new figure is created.
    :param figkwargs: kwargs for a new figure if no axes are provided.
    :return: figure and axes.
    """
    # create a new figure if no axes are provided
    if axs is None:
        try:
            figkwargs['figsize']
        except KeyError:
            figkwargs['figsize'] = (10, 5)
        fig, [ax1, ax2] = plt.subplots(1, 2, **figkwargs)
    else:
        ax1, ax2 = axs
        fig = ax1.figure
    # calculate phi and psi angles
    # currently only implemented for alanine dipeptide
    # TODO: make available for other molecules
    target_phi_psi = compute_phi_psi(val_traj)
    generated_phi_psi = compute_phi_psi(sample_traj)
    # plot the histograms
    plot_phi_psi(ax1, target_phi_psi)
    ax1.set_title("target")
    plot_phi_psi(ax2, generated_phi_psi)
    ax2.set_title("generated")
    return fig, [ax1, ax2]


def ramachandran_plot(val_dataset, topology, INN, latent_target_distribution, axs=None, **figkwargs):
    """
    Plot a Ramachandran plot of generated data compared to the validation data.
    :param val_dataset: The dataset that serves as comparison.
    :param topology: A mdtraj topology object that describes the molecule.
    :param INN: The invertible neural network that transforms the data.
    :param latent_target_distribution: The distribution q* in latent space.
    :param axs: The axes to plot on. If ``None``, a new figure is created.
    :param figkwargs: kwargs for a new figure if no axes are provided.
    :return: figure and axes.
    """
    # don't want to mess with the gradients while generating new samples --> no grad
    with torch.no_grad():
        n_samples = len(val_dataset)
        z = latent_target_distribution.sample((n_samples,))
        x = INN(z, rev=True)[0].cpu().detach().numpy().reshape(n_samples, -1, 3)
    val_tensor = val_dataset.get_tensor().cpu().detach().numpy().reshape(n_samples, -1, 3)
    # create mdtraj trajectories
    val_traj = mdtraj.Trajectory(val_tensor, topology)
    sample_traj = mdtraj.Trajectory(x, topology)
    # plot the ramachandran plots for both datasets
    fig, axs = ramachandran_plot_simple(val_traj, sample_traj, axs=axs, **figkwargs)
    return fig, axs


def mmd(x, y, kernels):
    """
    M = number of data samples
    N = number of syntethic samples
    D = Dimension of sample
    x is a MxD tensor
    y is a NxD tensor
    Don't make these too large!!

    kernels is a list of kernels that must accept square norm as only input
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, D = x.shape
    N, D2 = y.shape
    assert D == D2, "vector dims in x and y do not match."
    xx, xy, yy = torch.mm(x, x.t()), torch.mm(x, y.t()), torch.mm(y, y.t())
    # <x_i, x_j> MxM, <x_i, y_j> MxN, <y_i, y_j> NxN
    row_xx = xx.diag().unsqueeze(0).expand_as(xx)  # <x_i, x_i> in every row (diagonal of xx)  MxM
    row_yy = yy.diag().unsqueeze(0).expand_as(yy)  # <y_i, y_i> in every row (diagonal of yy)  NxN
    row_xy = xx.diag().unsqueeze(1).expand_as(xy)  # s.a., match shape of xy                   MxN
    row_yx = yy.diag().unsqueeze(0).expand_as(xy)  # MxN

    norm_xx = (row_xx + row_xx.t() - 2 * xx).to(device)
    norm_xx = norm_xx * torch.logical_not(torch.eye(M)).to(device)
    norm_yy = (row_yy + row_yy.t() - 2 * yy).to(device)
    norm_yy = norm_yy * torch.logical_not(torch.eye(N)).to(device)
    norm_xy = (row_xy + row_yx - 2 * xy).to(device)

    XX, YY, XY = torch.zeros_like(xx).to(device), torch.zeros_like(yy).to(device), torch.zeros_like(xy).to(device)
    for k in kernels:
        XX += k(norm_xx)
        YY += k(norm_yy)
        XY += k(norm_xy)
    return torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)


def sq_exp_kernel(sq_norm, sigma):
    return torch.exp(-.5 * sq_norm / sigma ** 2)


def rbf_generator(sigmata):
    return [lambda x: torch.exp(-.5 * x / s ** 2) for s in sigmata]


def multiscale_generator(sigmata):
    return [lambda x: s ** 2 / (x + s ** 2) for s in sigmata]
