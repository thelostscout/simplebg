from simplebg.model import ToyHParams, ToyExperiment
from simplebg.network.freia import RNVPConstWidthHParams
from simplebg.data.loader import ToyLoaderHParams
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights

Experiment = ToyExperiment

hparams = ToyHParams(
    loader_hparams=ToyLoaderHParams(
        name="MoonsDataset",
        samples=20_000,
    ),
    network_hparams=RNVPConstWidthHParams(
        coupling_blocks=12,
        subnet_hparams=dict(
            depth=6,
            width=32
        ),
    ),
    latent_hparams=DistributionHParams(
        name="Normal",
        kwargs={"sigma": 1.}
    ),
    loss_weights=LossWeights(
        nll_surrogate=1.
    ),
    max_epochs=200,
    batch_size=200,
    lr_scheduler="OneCycleLR",
    accelerator="auto",
)

trainer_kwargs = {"fast_dev_run": True}
logger_kwargs = {"name": "moons"}