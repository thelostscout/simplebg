import yaml
import sys
import os
import bgmol

from lightning_bg.architectures import *
from lightning_bg.utils import dataset_setter


def run_experiment(experiment_params, experiment_param_name, experiment_data_path):
    # load model and corresponding param classes
    ModelClass = get_network_by_name(experiment_params["network_name"])
    ParamClass = BaseHParams
    hparams = ParamClass(**experiment_params["network_params"])

    # import alanine data
    is_data_here = os.path.exists(experiment_data_path + "/Ala2TSF300.npy")
    ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=experiment_data_path)
    system = ala_data.system
    coordinates = ala_data.coordinates

    # prepare the data
    train_split = experiment_params["training_params"]["train_split"]
    train_data, val_data, test_data = dataset_setter(
        coordinates, system, val_split=(.8 - train_split), test_split=.2, seed=42
    )
    # create model
    if ModelClass.needs_energy_function:
        model = ModelClass(
            hparams, system.energy_model.energy, train_data=train_data, val_data=val_data
        )
    else:
        model = ModelClass(
            hparams, train_data=train_data, val_data=val_data
        )
    # load model state from previous experiment
    load_from_checkpoint = experiment_params.get("load_from_checkpoint", None)
    if load_from_checkpoint is not None:
        print(f"loading state_dict from data/lightning_logs/{load_from_checkpoint}/checkpoints/last.ckpt")
        checkpoint = torch.load(experiment_data_path + f"/lightning_logs/{load_from_checkpoint}/checkpoints/last.ckpt")
        model.load_state_dict(checkpoint['state_dict'])
    # fit the model
    model.fit(
        trainer_kwargs=experiment_params["trainer_kwargs"],
        logger_kwargs=dict(save_dir=experiment_data_path + f"lightning_logs", name=experiment_param_name)
    )
    pass


if __name__ == "__main__":
    # read yaml param file
    param_path = sys.argv[1]
    param_name = param_path[:-5]
    with open(param_path) as f:
        params = yaml.load(f, yaml.FullLoader)  # load the parameters

    # create a default datapath if none is given
    data_path = params.get("data_path", None)
    if not data_path:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.realpath(file_dir + "/../data/") + "/"
    else:
        data_path = os.path.expanduser(data_path)
        print("loading data path from yaml file: ", data_path)
        data_path = os.path.realpath(data_path) + "/"

    run_experiment(params, param_name, data_path)
