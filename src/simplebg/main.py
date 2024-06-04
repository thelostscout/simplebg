import sys
import lightning_trainable as lt
from simplebg.experiment.freia import ToyExperiment, PeptideExperiment, ToyHParams, PeptideHParams
from simplebg.network.freia import RNVPConstWidthHParams
from simplebg.data.loader import ToyLoaderHParams, PeptideLoaderHParams
from simplebg.latent import DistributionHParams
from simplebg.loss.core import LossWeights


# TODO: let the hparams be loaded from a params file
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
)

ala_hparams = PeptideHParams(
    loader_hparams=PeptideLoaderHParams(
        name="Ala2TSF300",
        root="../../data",
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
)


def main():
    model = ToyExperiment(hparams=hparams)
    lightning_logs = "../lightning_logs"
    name = "moons"
    model.fit(trainer_kwargs={}, logger_kwargs={"save_dir": lightning_logs, "name": name})


if __name__ == "__main__":
    main()
