from torch import nn

from simplebg.data import PeptideLoaderHParams
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights
from simplebg.model import PeptideHParams
from simplebg.network.freia import FixedBlocksHParams
from simplebg.network.fff import SubNetFreeFormFlowHParams
from simplebg.network.subnets import FullyConnectedHParams, ResNetHParams, NormalizerFreeResNetHParams

freia_network_hparams = FixedBlocksHParams(
    coupling_blocks=12,
    coupling_block_name="AllInOneBlock",
    subnet_hparams=ResNetHParams(
        depth_scheme=[3],
        width_scheme=[512],
        batch_norm=False,
        activation=nn.ReLU(),
    )
)

resnet_network_hparams = SubNetFreeFormFlowHParams(
    bottleneck=66,
    subnet_hparams=NormalizerFreeResNetHParams(
        depth_scheme=[10, 10, 10],
        width_scheme=[512, 256, 512],
        alpha=1.,
        scaled_weights=False,
    ),
    transform="ic",
)

hparams = PeptideHParams(
    loader_hparams=PeptideLoaderHParams(
        name="Ala2TSF300",
        root="ala2",
        method="bgmol",
        train_split=0.7,
        val_split=0.05,
        test_split=0.25,

    ),
    network_hparams=resnet_network_hparams,
    latent_hparams=DistributionHParams(
        name="Normal",
        kwargs={"sigma": 1.}
    ),
    loss_weights=LossWeights(
        forward_kl=1.,
        reconstruction=1_000.,
        reverse_kl=.001,
    ),
    max_epochs=200,
    batch_size=5000,
    lr_scheduler="OneCycleLR",
    optimizer=dict(
        name="Adam",
        lr=1.e-6,
        betas=[.99, .9999],
    ),
    accelerator="auto",
    gradient_clip=100.,
    # track_grad_norm=2,
)

trainer_kwargs = {"fast_dev_run": False, "enable_progress_bar": False}
logger_kwargs = {"name": "ala2"}
