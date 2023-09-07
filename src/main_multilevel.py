import yaml
import sys
import os
import bgmol
from ipdb import set_trace as bp

from lightning_bg.utils import dataset_setter
from manifold_learning import JointTrainableNetwork, PhaseTrainableNetwork, get_network_classes_by_name, \
    JointTrainableNetworkParams, PhaseTrainableNetworkParams, LossWeights
from manifold_learning.callbacks import PhaseSwitcher
import torch
torch.set_float32_matmul_precision('medium')

file_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.realpath(file_dir + "/../data/") + "/"


if __name__ == "__main__":
    param_path = sys.argv[1]
    with open(param_path) as f:
        params = yaml.load(f, yaml.FullLoader)  # load the parameters
    ModelClass, ParamClass = get_network_classes_by_name(
        params.get("network_type", "VanillaINN")
    )

    # import data
    is_data_here = os.path.exists(data_path + "Ala2TSF300.npy")
    ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=data_path)
    system = ala_data.system
    coordinates = ala_data.coordinates
    temperature = ala_data.temperature
    dim = ala_data.dim

    train_split = params["train_split"]
    test_split = .2
    val_split = 1 - train_split - test_split
    assert val_split > 0, f"'train_split' has to be smaller than {1 - test_split}"
    train_data, val_data, test_data = dataset_setter(
        coordinates, system, val_split=(1 - test_split - train_split), test_split=test_split, seed=42
    )


    def dataset(labels=False):
        return dict(train=train_data, val=val_data, test=test_data, shift=0, scale=1, dims_cond=[], dims=[66,])

    bp()
    params["training_params"]["dataset"] = dataset

    # Create TrainableNetwork hyperparameters
    params["training_params"]["network_params"] = ParamClass(
        **params.get("network_params", {})
    )
    for k, v in params["training_params"].items():
        if k in ["joint_loss", "wavelet_loss", "detail_loss", "coarse_loss"]:
            # TODO: allow loss list
            params["training_params"][k] = LossWeights(**v)
    params["training_params"]["network_type"] = ModelClass
    trainer_kwargs = params.get("trainer_kwargs", {})
    training_type = params.get("training_type", "joint" if ModelClass.__name__ != "MultilevelINN" else "phase")
    if training_type == 'joint':
        training_params = JointTrainableNetworkParams(**params["training_params"])
        model = JointTrainableNetwork(training_params)
        callbacks = []
    elif training_type == 'phase':
        training_params = PhaseTrainableNetworkParams(**params["training_params"])
        model = PhaseTrainableNetwork(training_params)
        callbacks = [PhaseSwitcher()]
    else:
        raise ValueError(f"training_type must be 'phase' or 'joint'")
    # Fit and plot the model
    trainer_kwargs["callbacks"] = callbacks
    bp()
    model.fit(
        logger_kwargs={"save_dir": data_path + "lightning_logs/", "name": params["name"]},
        trainer_kwargs=trainer_kwargs,
    )
    print("Experiment complete \U0001F389")
