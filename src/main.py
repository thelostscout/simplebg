import yaml
import sys
import os
import bgmol

from lightning_bg.architectures import *
from lightning_bg.utils import dataset_setter

if __name__ == "__main__":
    param_path = sys.argv[1]
    param_name = param_path[:-5]
    with open(param_path) as f:
        params = yaml.load(f, yaml.FullLoader)  # load the parameters

    data_path = params.get("data_path", None)
    if not data_path:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.realpath(file_dir + "/../data/") + "/"
    else:
        data_path = os.path.realpath(data_path) + "/"

    ModelClass = get_network_by_name(params["network_name"])
    ParamClass = BaseHParams
    hparams = ParamClass(**params["network_params"])

    # import data
    is_data_here = os.path.exists(data_path+"/Ala2TSF300.npy")
    ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=data_path)
    system = ala_data.system
    coordinates = ala_data.coordinates
    temperature = ala_data.temperature
    dim = ala_data.dim

    train_split = params["training_params"]["train_split"]
    # train_splits = [.7, .1, .01, .001]
    # for train_split in train_splits:
    train_data, val_data, test_data = dataset_setter(
        coordinates, system, val_split=(.8-train_split), test_split=.2, seed=42
    )
    if ModelClass.needs_energy_function:
        model = ModelClass(
            hparams, system.energy_model.energy, train_data=train_data, val_data=val_data
        )
    else:
        model = ModelClass(
            hparams, train_data=train_data, val_data=val_data
        )
    load_from_checkpoint = params.get("load_from_checkpoint", None)
    if load_from_checkpoint is not None:
        print(f"loading state_dict from data/lightning_logs/{load_from_checkpoint}/checkpoints/last.ckpt")
        checkpoint = torch.load(data_path+f"/lightning_logs/{load_from_checkpoint}/checkpoints/last.ckpt")
        model.load_state_dict(checkpoint['state_dict'])
    model.fit(
        trainer_kwargs=params["trainer_kwargs"],
        logger_kwargs=dict(save_dir=data_path+f"lightning_logs", name=param_name)
    )
