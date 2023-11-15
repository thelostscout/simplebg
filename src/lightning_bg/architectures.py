from functools import partial

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import lightning_trainable as lt
import torch
import torch.distributions as D
import torch.nn as nn
from FrEIA.utils import force_to
from abc import ABC, abstractmethod

from lightning_trainable.hparams import AttributeDict

from lightning_bg.utils import Alignment


def get_network_by_name(name: str):
    if name == "RNVPfwkl":
        return RNVPfwkl
    elif name == "RNVPpseudofwkl":
        return RNVPpseudofwkl
    elif name == "RNVPvar":
        return RNVPvar
    elif name == "RNVPrvkl":
        return RNVPrvkl
    elif name == "RNVPrvklLatent":
        return RNVPrvklLatent
    elif name == "RQSfwkl":
        return RQSfwkl
    else:
        raise ValueError(f"Unknown network name {name}")


def latent_distribution_constructor(n_dims, **kwargs):
    name = kwargs['name']

    if name == "Normal":
        sigma = kwargs['sigma']
        return D.MultivariateNormal(torch.zeros(n_dims), sigma * torch.eye(n_dims))
    elif name == "Bimodal":
        sigmas = torch.tensor(kwargs['sigmas'])[:, None] * torch.eye(n_dims)
        mus = torch.zeros((2, n_dims))
        mus[0, 0], mus[1, 0] = kwargs['mus']
        gausses = D.MultivariateNormal(mus, sigmas)
        weights = D.Categorical(torch.tensor([1, 1]))
        return D.MixtureSameFamily(weights, gausses)
    else:
        raise ValueError(f"Unknown distribution name {name}")


def exponential_subnet_constructor(subnet_max_width, subnet_depth, subnet_growth_factor, dims_in, dims_out):
    assert subnet_max_width >= dims_in
    dims_prev = dims_in
    layers = []
    for i in range(subnet_depth + 1):
        if i < subnet_depth / 2:
            dims_next = dims_prev * subnet_growth_factor
        elif i == subnet_depth / 2:
            dims_next = dims_prev
        else:
            dims_next = int(dims_prev / subnet_growth_factor)
        if i != subnet_depth:
            layers.append(nn.Linear(min(dims_prev, subnet_max_width), min(dims_next, subnet_max_width)))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(min(dims_prev, subnet_max_width), dims_out))
        dims_prev = dims_next
    block = nn.Sequential(*layers)
    block[-1].weight.data.zero_()
    block[-1].bias.data.zero_()
    return block


class BaseHParams(lt.TrainableHParams):
    inn_depth: int
    subnet_max_width: int
    subnet_depth: int
    subnet_growth_factor: int = 2
    n_dims: int
    latent_target_distribution: dict


class RvklHParams(BaseHParams):
    is_molecule: bool = True
    lambda_alignment: float | None = None


class RvklLatentHParams(RvklHParams):
    latent_network_params: BaseHParams

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams.latent_network_params.n_dims = hparams.n_dims
        return super().validate_parameters(hparams)


class BaseTrainable(lt.Trainable, ABC):
    hparams: BaseHParams
    needs_energy_function = False

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.inn = self.configure_inn()
        self.q = latent_distribution_constructor(self.hparams.n_dims, **self.hparams.latent_target_distribution)

        print(self.hparams.max_epochs)
        print(self.hparams.optimizer)

    @abstractmethod
    def configure_inn(self):
        raise NotImplementedError

    def sample(self, size):
        with torch.no_grad():
            z = self.q.sample(size)
            x = self.inn(z, rev=True)[0]
        return x

    def log_prob(self, x):
        z, log_det_JF = self.inn(x)
        return self.q.log_prob(z) + log_det_JF

    def to(self, *args, **kwargs):
        if isinstance(self.q, TrainableDistribution):
            self.q.model.to(*args, **kwargs)
        elif isinstance(self.q, D.Distribution):
            force_to(self.q, *args, **kwargs)
        else:
            raise ValueError(f"Unknown distribution type {type(self.q)} for latent distribution.")
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


class TrainableDistribution(D.Distribution):
    def __init__(self, model: BaseTrainable):
        self.model = model
        super().__init__(validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.model.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.model.sample(sample_shape)


class BaseRNVP(BaseTrainable):
    def configure_inn(self):
        inn = Ff.SequenceINN(self.hparams.n_dims)
        # normalize inputs
        inn.append(Fm.ActNorm)
        # add coupling blocks
        for k in range(self.hparams.inn_depth):
            inn.append(
                Fm.RNVPCouplingBlock,
                subnet_constructor=partial(
                    exponential_subnet_constructor,
                    self.hparams.subnet_max_width,
                    self.hparams.subnet_depth,
                    self.hparams.subnet_growth_factor,
                )
            )
        return inn


class BaseRNVPEnergy(BaseRNVP):
    needs_energy_function = True
    needs_alignment = False

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


class RNVPpseudofwkl(BaseRNVPEnergy):
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
    hparams: RvklHParams
    needs_alignment = True

    def __init__(self, hparams, energy_function, alignment_penalty: Alignment.penalty, **kwargs):
        super().__init__(hparams, energy_function, **kwargs)
        self.is_molecule = self.hparams.is_molecule
        self.alignment_penalty = alignment_penalty

    def rvkl_loss(self, z):
        xG, log_det_JG = self.inn(z, rev=True)
        log_pB = self.pB_log_prob(xG)
        if self.is_molecule:
            alignment_penalty = self.alignment_penalty(xG) * self.hparams.lambda_alignment
            alignment_penalty = alignment_penalty.to(self.device)
        else:
            alignment_penalty = 0

        return - log_pB - log_det_JG + alignment_penalty

    def compute_metrics(self, batch, batch_idx) -> dict:
        # noinspection PyTypeChecker
        z = self.q.sample((self.hparams.batch_size,))
        loss = self.rvkl_loss(z)

        return dict(
            loss=loss.mean(),
            # alignment_penalty=aligment_penalty.mean(),
        )


class RNVPrvklLatent(RNVPrvkl):
    hparams: RvklLatentHParams

    def __init__(self, hparams, energy_function, alignment_penalty: Alignment.penalty, **kwargs):
        super().__init__(hparams, energy_function, alignment_penalty, **kwargs)
        model = RNVPfwkl(hparams.latent_network_params, **kwargs)
        self.q = TrainableDistribution(model)

    def fit(self, logger_kwargs: dict = None, trainer_kwargs: dict = None, fit_kwargs: dict = None) -> dict:
        # set latent name in logger
        if logger_kwargs is not None:
            latent_logger_kwargs = logger_kwargs.copy()
        else:
            latent_logger_kwargs = dict()
        latent_logger_kwargs["sub_dir"] = "latent"
        # train latent network
        # for p in self.inn.parameters():
        #     p.requires_grad = False
        # self.inn.eval()
        # for pq in self.q.parameters():
        #     pq.requires_grad = True
        # self.q.train()
        # self.q.fit(latent_logger_kwargs, trainer_kwargs, fit_kwargs)
        # for pq in self.q.parameters():
        #     pq.requires_grad = False
        # self.q.eval()
        # for p in self.inn.parameters():
        #     p.requires_grad = True
        # self.inn.train()
        self.q.model.fit(latent_logger_kwargs, trainer_kwargs, fit_kwargs)
        return super().fit(logger_kwargs, trainer_kwargs, fit_kwargs)


class BaseRQS(BaseTrainable):
    def configure_inn(self):
        subnet_width = self.hparams.subnet_width
        inn_depth = self.hparams.inn_depth
        n_dims = self.hparams.n_dims
        inn = Ff.SequenceINN(n_dims)
        inn.append(Fm.ActNorm)
        for k in range(inn_depth):
            inn.append(Fm.RationalQuadraticSpline, subnet_constructor=partial(self.subnet_constructor, subnet_width))
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


class RQSfwkl(BaseRQS):
    def forward_kl_loss(self, z, log_det_JF):
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx):
        z, log_det_JF = self.inn(batch)
        loss = self.forward_kl_loss(z, log_det_JF).mean()
        return dict(
            loss=loss,
        )
