from simplebg.data import PeptideLoaderHParams
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights
from simplebg.model import PeptideHParams as RNPeptideHParams, PeptideExperiment as RNPeptideExperiment
from simplebg.network.freia import RNVPConstWidthHParams
from simplebg.network.resnet import ResNetHParams
from simplebg.network.subnets import ConstWidthHParams

Experiment = RNPeptideExperiment

freia_network_hparams = RNVPConstWidthHParams(
    coupling_blocks=12,
    subnet_hparams=ConstWidthHParams(
        depth=6,
        width=256,
    )),

resnet_network_hparams = ResNetHParams(
    bottleneck=66,
    depth=40,
    width=512,
    dropout=.05,
    residual=True
),

hparams = RNPeptideHParams(
    loader_hparams=PeptideLoaderHParams(
        name="Ala2TSF300",
        root="../../data/",
        method="bgmol",
    ),
    network_hparams=resnet_network_hparams,
    latent_hparams=DistributionHParams(
        name="Normal",
        kwargs={"sigma": 1.}
    ),
    loss_weights=LossWeights(
        nll_surrogate=1.
    ),
    max_epochs=20,
    batch_size=1000,
    lr_scheduler="OneCycleLR",
    accelerator="auto",
)

trainer_kwargs = {"fast_dev_run": False}
logger_kwargs = {"name": "ala2"}
