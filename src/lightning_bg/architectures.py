from functools import partial

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import lightning_trainable as lt
import torch
import torch.distributions as D
import torch.nn as nn
from FrEIA.utils import force_to


def get_network_by_name(name: str):
    if name == "RNVPfwkl":
        return RNVPfwkl
    elif name == "RNVPartfwkl":
        return RNVPartfwkl
    elif name == "RNVPvar":
        return RNVPvar
    elif name == "RNVPrvkl":
        return RNVPrvkl
    else:
        raise ValueError(f"Unknown network name {name}")


class BaseHParams(lt.TrainableHParams):
    inn_depth: int
    subnet_width: int
    n_dims: int
    latent_target_distribution: dict


class BaseRNVP(lt.Trainable):
    hparams: BaseHParams
    needs_energy_function = False

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.inn = self.configure_inn()
        self.q = self.latent_distribution_constructor(**hparams.latent_target_distribution)

    def latent_distribution_constructor(self, **kwargs):
        n_dims = self.hparams.n_dims
        name = kwargs['name']

        if name == "Normal":
            sigma = kwargs['sigma']
            return D.MultivariateNormal(torch.zeros(n_dims), sigma * torch.eye(n_dims))

    def configure_inn(self):
        subnet_width = self.hparams.subnet_width
        inn_depth = self.hparams.inn_depth
        n_dims = self.hparams.n_dims
        inn = Ff.SequenceINN(n_dims)
        inn.append(Fm.ActNorm)
        for k in range(inn_depth):
            inn.append(Fm.RNVPCouplingBlock, subnet_constructor=partial(self.subnet_constructor, subnet_width))
        return inn

    @staticmethod
    def subnet_constructor(subnet_width, dims_in, dims_out):
        block = nn.Sequential(nn.Linear(dims_in, subnet_width), nn.ReLU(),
                              nn.Linear(subnet_width, subnet_width), nn.ReLU(),
                              nn.Linear(subnet_width, subnet_width), nn.ReLU(),
                              nn.Linear(subnet_width, dims_out))

        block[-1].weight.data.zero_()
        block[-1].bias.data.zero_()
        return block

    def generate_samples(self, size):
        with torch.no_grad():
            z = self.q.sample(size)
            x = self.inn(z, rev=True)[0]
        return x

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


class BaseRNVPEnergy(BaseRNVP):
    needs_energy_function = True

    def __init__(self, hparams, energy_function, **kwargs):
        super().__init__(hparams, **kwargs)
        self.pB_log_prob = lambda x: - energy_function(x)
        pass


class RNVPfwkl(BaseRNVP):
    hparams: BaseHParams

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

    def forward_kl_loss(self, z, log_det_JF):
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx):
        z, log_det_JF = self.inn(batch)
        loss = self.forward_kl_loss(z, log_det_JF).mean()
        return dict(
            loss=loss,
        )


class RNVPartfwkl(BaseRNVPEnergy):
    hparams: BaseHParams

    def art_fwkl_loss(self, xG):
        z, log_det_JF = self.inn(xG)
        log_pB = self.pB_log_prob(xG)
        log_pG = self.q.log_prob(z) + log_det_JF
        with torch.no_grad():
            reweight = torch.exp(log_pB - log_pG)
        return (reweight * log_pG).mean()

    def compute_metrics(self, batch, batch_idx):
        with torch.no_grad():
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            xG = self.inn(z, rev=True)[0]
        loss = self.art_fwkl_loss(xG)
        return dict(
            loss=loss,
            energy=self.pB_log_prob(xG).mean().cpu().detach()
        )


class RNVPvar(BaseRNVPEnergy):
    hparams: BaseHParams

    def var_loss(self, xG):
        z, log_det_JF = self.inn(xG)
        log_pB = self.pB_log_prob(xG)
        log_pG = self.q.log_prob(z) + log_det_JF
        log_ratios = log_pB - log_pG
        with torch.no_grad():
            K = log_ratios.mean()
        return torch.nn.functional.relu(log_ratios - K).square().mean()

    def compute_metrics(self, batch, batch_idx):
        with torch.no_grad():
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            xG = self.inn(z, rev=True)[0]
        # log_det_JG = self.inn(z, rev=True)[1]

        loss = self.var_loss(xG)
        return dict(
            loss=loss,
            energy=self.pB_log_prob(xG).mean().cpu().detach()
        )


class RNVPfrkl(BaseRNVPEnergy):
    hparams: BaseHParams

    def fwkl_loss(self, z, log_det_JF):
        log_likelihood = self.q.log_prob(z) + log_det_JF  # log_det_JF = - log_det_JG
        return - log_likelihood

    def rvkl_loss(self, xG):
        z, log_det_JF = self.inn(xG)
        log_pB = - self.energy_function(xG)
        log_pG = self.q.log_prob(z) + log_det_JF
        return log_pG - log_pB

    def compute_metrics(self, batch, batch_idx) -> dict:
        if batch_idx % 2:
            z, log_det_JF = self.inn(batch)
            loss = self.fwkl_loss(z, log_det_JF).mean()
        else:
            with torch.no_grad():
                # noinspection PyTypeChecker
                z = self.q.sample((self.hparams.batch_size,))
                xG = self.inn(z, rev=True)[0]
            loss = self.rvkl_loss(xG).mean()

        return dict(
            loss=loss,
        )


class RNVPrvkl(BaseRNVPEnergy):
    hparams: BaseHParams

    def rvkl_loss(self, z):
        xG, log_det_JG = self.inn(z, rev=True)
        log_pB = - self.pB_log_prob(xG)
        log_pG = self.q.log_prob(z) - log_det_JG
        return log_pG - log_pB

    def left_side_ratio(self, z):
        xG = self.inn(z, rev=True)[0]
        return (xG[:, 0] < 5).float().mean()

    # noinspection PyTypeChecker
    def compute_metrics(self, batch, batch_idx) -> dict:
        z = self.q.sample((self.hparams.batch_size,))
        loss = self.rvkl_loss(z).mean()
        ratio = self.left_side_ratio(z)

        return dict(
            loss=loss,
            left_side_ratio=ratio
        )
