from simplebg.data import PeptideLoaderHParams
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights
from simplebg.model import PeptideHParams
from simplebg.network.freia import RNVPConstWidthHParams
from simplebg.network.fff import ResNetHParams, ConstWidthHParams as ResNetConstWidthHParams
from simplebg.network.subnets import ConstWidthHParams

freia_network_hparams = RNVPConstWidthHParams(
    coupling_blocks=12,
    subnet_hparams=ConstWidthHParams(
        depth=3,
        width=512,
        block_depth=2,
        dropout=0.,
        residual=True,
    )
)

resnet_network_hparams = ResNetHParams(
    bottleneck=66,
    net_hparams=ResNetConstWidthHParams(
        depth=31,
        width=512,
        block_depth=2,
        dropout=0.,
        residual=True,
    ),
    transform="ic",
    transform_kwargs=dict(
        normalize_angles=True
    )
)

hparams = PeptideHParams(
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
        forward_kl=1.,
        reconstruction=5_000.,
    ),
    max_epochs=20,
    batch_size=5000,
    lr_scheduler="OneCycleLR",
    optimizer=dict(
        name="Adam",
        lr=1.e-6,
        betas=[.99, .9999],
    ),
    accelerator="auto",
    track_grad_norm=2,
    gradient_clip=1_000.,
)

trainer_kwargs = {"fast_dev_run": False, "enable_progress_bar": False}
logger_kwargs = {"name": "ala2"}
