from simplebg.data import PeptideLoaderHParams
from simplebg.experiment.freia import PeptideHParams, PeptideExperiment
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights
from simplebg.network.freia import RNVPConstWidthHParams

Experiment = PeptideExperiment

hparams = PeptideHParams(
    loader_hparams=PeptideLoaderHParams(
        name="Ala2TSF300",
        root="../../data/",
        method="bgmol",
    ),
    network_hparams=RNVPConstWidthHParams(
        coupling_blocks=12,
        subnet_hparams=dict(
            depth=6,
            width=128
        ),
    ),
    latent_hparams=DistributionHParams(
        name="Normal",
        kwargs={"sigma": 1.}
    ),
    loss_weights=LossWeights(
        forward_kl=1.
    ),
    max_epochs=20,
    batch_size=1000,
    lr_scheduler="OneCycleLR",
    accelerator="auto",
)

trainer_kwargs = {"fast_dev_run": False}
logger_kwargs = {"name": "ala2"}