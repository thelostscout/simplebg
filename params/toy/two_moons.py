from simplebg.experiment.freia import ToyHParams
from simplebg.network.freia import RNVPConstWidthHParams
from simplebg.data.loader import ToyLoaderHParams, SplitHParams
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights

hparams = ToyHParams(
    loader_hparams=ToyLoaderHParams(
        name="MoonsDataset",
        max_samples=20_000
    ),
    split_hparams=SplitHParams(),
    network_hparams=RNVPConstWidthHParams(
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
        forward_kl=1.
    ),
    max_epochs=20,
    batch_size=200,
    lr_scheduler="OneCycleLR",
)
