import simplebg

network_hparams = simplebg.model.NetworkHParams(
    nontrivial_torsions_depth=6,
    trivial_torsions_depth=6,
    angles_depth=4,
    bonds_depth=4,
)

loss_weights = simplebg.loss.core.LossWeights(
    forward_kl=1.,
    reconstruction=0.,
)

loader_hparams = simplebg.data.PeptideLoaderHParams(
    name="Ala2TSF300",
    root="ala2",
    method="bgmol",
    train_split=0.7,
    val_split=0.05,
    test_split=0.25,
)

latent_hparams = simplebg.latent.DistributionHParams(
    name="Normal",
    kwargs={"sigma": 1.}
)

hparams = simplebg.model.FlowHParams(
    model_class="FlowModel",
    shapiro_threshold=1e-30,
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
    loader_hparams=loader_hparams,
    network_hparams=network_hparams,
    latent_hparams=latent_hparams,
    loss_weights=loss_weights,
)

trainer_kwargs = {"fast_dev_run": False}
