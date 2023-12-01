from functools import partial
import os
from typing import Optional

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import lightning_trainable as lt
import torch
import torch.distributions as D
import torch.nn as nn
from FrEIA.utils import force_to
from abc import ABC, abstractmethod

from lightning_trainable.hparams import AttributeDict

from .utils import AlignmentIC, AlignmentRMS, ICTransform


def get_network_by_name(name: str):
    if name == "RNVPfwkl":
        return RNVPfwkl
    elif name == "RNVPICfwkl":
        return RNVPICfwkl
    elif name == "RNVPpseudofwkl":
        return RNVPpseudofwkl
    elif name == "RNVPvar":
        return RNVPvar
    elif name == "RNVPrvkl":
        return RNVPrvkl
    elif name == "RNVPfwrvkl":
        return RNVPfwrvkl
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
    needs_system = False

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.inn = self.configure_inn()
        self.q = latent_distribution_constructor(self.hparams.n_dims, **self.hparams.latent_target_distribution)

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


class BaseRNVPIC(BaseTrainable):
    needs_system = True

    def __init__(self, hparams, system=None, **kwargs):
        if system is None:
            raise ValueError("System must be provided for RNVPIC.")
        self.system = system
        super().__init__(hparams, **kwargs)

    def configure_inn(self):
        inn = Ff.SequenceINN(self.hparams.n_dims)
        # transform to normalised internal coordinates
        inn.append(ICTransform, system=self.system)
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


class RNVPICfwkl(BaseRNVPIC):
    def forward_kl_loss(self, z, log_det_JF):
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx):
        z, log_det_JF = self.inn(batch)
        loss = self.forward_kl_loss(z, log_det_JF).mean()
        return dict(
            loss=loss,
        )


class BaseRNVPEnergy(BaseRNVP):
    needs_energy_function = True
    needs_alignment = False

    def __init__(self, hparams, energy_function=None, **kwargs):
        if energy_function is None:
            raise ValueError("Energy function must be provided for RNVPEnergy.")
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

    def pseudo_forward_kl_loss(self, xG):
        z, log_det_JF = self.inn(xG)
        log_pB = self.pB_log_prob(xG)
        log_pG = self.q.log_prob(z) + log_det_JF
        with torch.no_grad():
            print(log_pB.shape)
            reweight = log_pB.softmax(dim=0) / (log_pG.softmax(dim=0) + 1e-9)
        return reweight * log_pG, reweight

    def compute_metrics(self, batch, batch_idx):
        with torch.no_grad():
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            xG = self.inn(z, rev=True)[0]
        loss, reweight = self.pseudo_forward_kl_loss(xG)
        return dict(
            loss=loss.mean(),
            reweight=reweight.mean(),
            energy=self.pB_log_prob(xG).mean().cpu().detach()
        )


class RNVPrvkl(BaseRNVPEnergy):
    hparams: RvklHParams
    needs_alignment = True

    def __init__(self, hparams, energy_function=None, alignment_penalty=None, **kwargs):
        if alignment_penalty is None:
            raise ValueError("Alignment penalty must be provided for RNVPrvkl.")
        super().__init__(hparams, energy_function, **kwargs)
        self.is_molecule = self.hparams.is_molecule
        self.alignment_penalty = alignment_penalty

    def reverse_kl_loss(self, z):
        xG, log_det_JG = self.inn(z, rev=True)
        log_pB = self.pB_log_prob(xG)
        if self.is_molecule:
            alignment_penalty_loc, alignment_penalty_rot = self.alignment_penalty(xG)
            alignment_penalty_loc, alignment_penalty_rot = alignment_penalty_loc.to(
                self.device) * self.hparams.lambda_alignment, alignment_penalty_rot.to(
                self.device) * self.hparams.lambda_alignment
        else:
            alignment_penalty_loc, alignment_penalty_rot = 0, 0

        lambda_alignment_rot = 1.
        return - log_pB - log_det_JG + alignment_penalty_loc + lambda_alignment_rot * alignment_penalty_rot, log_pB, log_det_JG, alignment_penalty_loc, alignment_penalty_rot

    def compute_metrics(self, batch, batch_idx) -> dict:
        # noinspection PyTypeChecker
        z = self.q.sample((self.hparams.batch_size,))
        loss, log_pB, log_det_JG, alignment_penalty_loc, alignment_penalty_rot = self.reverse_kl_loss(z)

        return dict(
            loss=loss.mean(),
        )


class RNVPfwrvkl(RNVPrvkl):
    def forward_kl_loss(self, z, log_det_JF):
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx) -> dict:
        if batch_idx // 2:
            z, log_det_JF = self.inn(batch)
            loss = self.forward_kl_loss(z, log_det_JF)
            return dict(
                loss=loss.mean(),
            )
        else:
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            loss = self.reverse_kl_loss(z)[0]
            return dict(
                loss=loss.mean(),
            )


class RNVPvar(RNVPrvkl):

    def var_loss(self, xG):
        z, log_det_JF = self.inn(xG)
        log_pB = self.pB_log_prob(xG)
        log_pG = self.q.log_prob(z) + log_det_JF
        log_ratios = log_pB - log_pG
        with torch.no_grad():
            K = log_ratios.mean()

        if self.is_molecule:
            alignment_penalty_loc, alignment_penalty_rot = self.alignment_penalty(xG)
            alignment_penalty_loc, alignment_penalty_rot = alignment_penalty_loc.to(
                self.device) * self.hparams.lambda_alignment, alignment_penalty_rot.to(
                self.device) * self.hparams.lambda_alignment
        else:
            alignment_penalty_loc, alignment_penalty_rot = 0, 0
        return torch.nn.functional.relu(log_ratios - K).square() + alignment_penalty_loc + alignment_penalty_rot

    def compute_metrics(self, batch, batch_idx):
        with torch.no_grad():
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            xG = self.inn(z, rev=True)[0]

        loss = self.var_loss(xG)
        return dict(
            loss=loss.mean(),
            energy=self.pB_log_prob(xG).mean().cpu().detach()
        )


class RNVPrvklLatent(RNVPrvkl):
    hparams: RvklLatentHParams

    def __init__(self, hparams, energy_function, alignment_penalty: AlignmentRMS.penalty, **kwargs):
        super().__init__(hparams, energy_function, alignment_penalty, **kwargs)
        if hparams.latent_network_params.model_checkpoint is None:
            hparams.latent_network_params.model_checkpoint = dict(dirpath="latent")
        else:
            hparams.latent_network_params.model_checkpoint["dirpath"] = "latent"
        model = RNVPfwkl(hparams.latent_network_params, **kwargs)
        self.q = TrainableDistribution(model)

    def fit(self, logger_kwargs: dict = None, trainer_kwargs: dict = None, fit_kwargs: dict = None) -> dict:
        # set latent name in logger
        if logger_kwargs is not None:
            latent_logger_kwargs = logger_kwargs.copy()
        else:
            latent_logger_kwargs = dict()
        latent_logger_kwargs["sub_dir"] = "latent"

        # make sure that version number is the same accross latent and main training
        if logger_kwargs.get("version", None) is None:
            version_number = self.find_next_version(logger_kwargs["save_dir"], logger_kwargs["name"])
            latent_logger_kwargs["version"] = f"version_{version_number}"
            logger_kwargs["version"] = f"version_{version_number}"
        self.q.model.fit(latent_logger_kwargs, trainer_kwargs, fit_kwargs)
        return super().fit(logger_kwargs, trainer_kwargs, fit_kwargs)

    @staticmethod
    def find_next_version(lightning_logs, param_name):
        dir_path = os.path.join(lightning_logs, param_name)
        dirs = os.listdir(dir_path)
        dirs_with_version = [f for f in dirs if f.startswith("version_")]
        if not dirs_with_version:
            return 0
        else:
            versions = [int(f.split("_")[1]) for f in dirs_with_version]
            return max(versions) + 1


class BaseRQS(BaseTrainable):
    def configure_inn(self):
        inn = Ff.SequenceINN(self.hparams.n_dims)
        # normalize inputs
        inn.append(Fm.ActNorm)
        # add coupling blocks
        for k in range(self.hparams.inn_depth):
            inn.append(
                Fm.RationalQuadraticSpline,
                subnet_constructor=partial(
                    exponential_subnet_constructor,
                    self.hparams.subnet_max_width,
                    self.hparams.subnet_depth,
                    self.hparams.subnet_growth_factor,
                )
            )
        return inn


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
