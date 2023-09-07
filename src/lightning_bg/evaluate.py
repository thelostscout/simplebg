import ipywidgets as ipw
import mdtraj
import nglview
import numpy as np
import torch
import matplotlib.pyplot as plt
from bgmol.systems.ala2 import compute_phi_psi
import matplotlib as mpl


class ShowTrajSimple(ipw.VBox):
    def __init__(self, traj, energies, **kwargs):
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

    def on_value_change(self, change):
        self.wnumber.value = np.around(self.energies[change['new']].item(), decimals=2)


class ShowTraj(ShowTrajSimple):
    def __init__(self, samples, system, **kwargs):
        traj = self.create_traj(samples, system)
        energies = system.energy_model.energy(samples)

        super().__init__(traj, energies, **kwargs)

    @staticmethod
    def create_traj(samples, system):
        samples_flat = samples.reshape(len(samples), -1, 3).cpu().detach().numpy()
        return mdtraj.Trajectory(samples_flat, system.mdtraj_topology)


class Evaluator:
    def __init__(self, model, system):
        self.inn = model.inn
        self.val_data = model.val_data
        self.system = system
        self.q = model.q
        self._all_funcs = [self.energy_plot, self.ramachandran_plot]

    def energy_plot(self, rg=None, **figkwargs):
        if rg is None:
            rg = [-50, 100]
        return energy_plot(self.val_data, self.system.energy_model.energy, self.inn, self.q, rg,
                           **figkwargs)

    def ramachandran_plot(self, **figkwargs):
        return ramachandran_plot(self.val_data, self.system.mdtraj_topology, self.inn, self.q,
                                 **figkwargs)

    def plot_all(self, **figkwargs):
        outputs = []
        for f in self._all_funcs:
            # noinspection PyArgumentList
            out = f(**figkwargs)
            outputs.append(out)
        return outputs


def energy_plot_simple(val_energies, sample_energies, rg, **figkwargs):
    fig = plt.figure(**figkwargs)
    ax = plt.gca()
    ax.hist(val_energies.cpu().detach().numpy(), bins=200, histtype='step', range=rg, label="target")
    ax.hist(sample_energies.cpu().detach().numpy(), bins=200, histtype='step', range=rg, label="generated")
    ax.set_yscale('log')
    ax.set_ylabel("# samples")
    ax.set_xlabel(r"Energy [$k_B T$]")
    ax.legend()
    return fig, ax


def energy_plot(val_dataset, energy_function, INN, latent_target_distribution, rg, **figkwargs):
    with torch.no_grad():
        n_samples = len(val_dataset)
        z = latent_target_distribution.sample((n_samples,))
        x = INN(z, rev=True)[0]
        energies = energy_function(x)
        val_tensor = val_dataset.get_tensor()
        val_energies = energy_function(val_tensor)
    fig, ax = energy_plot_simple(val_energies, energies, rg, **figkwargs)
    return fig, ax


# TODO: add plots for marginals of bonds etc. or different force term contributions

def plot_phi_psi(ax, phi_psi, bins=100):
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    ax.hist2d(phi_psi[0], phi_psi[1], bins=bins, density=True, norm=mpl.colors.LogNorm())


def ramachandran_plot_simple(val_traj, sample_traj, **figkwargs):
    try:
        figkwargs['figsize']
    except KeyError:
        figkwargs['figsize'] = (10, 5)
    target_phi_psi = compute_phi_psi(val_traj)
    generated_phi_psi = compute_phi_psi(sample_traj)

    fig, [ax1, ax2] = plt.subplots(1, 2, **figkwargs)
    plot_phi_psi(ax1, target_phi_psi)
    plt.suptitle("Ramachandran Plot")
    ax1.set_title("target")
    plot_phi_psi(ax2, generated_phi_psi)
    ax2.set_title("generated")
    return fig, [ax1, ax2]


def ramachandran_plot(val_dataset, topology, INN, latent_target_distribution, **figkwargs):
    with torch.no_grad():
        n_samples = len(val_dataset)
        z = latent_target_distribution.sample((n_samples,))
        x = INN(z, rev=True)[0].cpu().detach().numpy().reshape(n_samples, -1, 3)
    val_tensor = val_dataset.get_tensor().cpu().detach().numpy().reshape(n_samples, -1, 3)

    val_traj = mdtraj.Trajectory(val_tensor, topology)
    sample_traj = mdtraj.Trajectory(x, topology)
    fig, axs = ramachandran_plot_simple(val_traj, sample_traj, **figkwargs)
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
