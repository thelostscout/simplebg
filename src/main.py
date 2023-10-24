import os
import sys

import bgmol
from bgmol.systems.peptide import peptide
import mdtraj
import yaml

from lightning_bg.architectures import *
from lightning_bg.utils import dataset_setter


def run_experiment(experiment_params, experiment_param_name, experiment_data_path):
    # set log path
    lightning_logs = os.path.join(experiment_data_path, "lightning_logs", experiment_params['molecule'].lstrip("/"))
    # import data
    molecule_path = os.path.join(experiment_data_path, "Molecules", params['molecule'].lstrip("/"))
    if experiment_params['molecule'] == "Dialanine":
        # import alanine data
        is_data_here = os.path.exists(molecule_path + "/Ala2TSF300.npy")
        ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=molecule_path)
        # define system & energy model
        system = ala_data.system
        energy_model = system.reinitialize_energy_model(temperature=300., n_workers=1)
        coordinates = ala_data.coordinates
    else:
        # read the top.pdb
        with open(molecule_path.rstrip("/") + "/top.pdb", 'r') as file:
            lines = file.readlines()
            lastline = lines[-3]
            n_atoms = int(lastline[4:11].strip())
            n_res = int(lastline[22:26].strip())
            print(f"Number of atoms: {n_atoms}, residues: {n_res}")

        # define system & energy model
        system = peptide(short=False, n_atoms=n_atoms, n_res=n_res, filepath=molecule_path)
        energy_model = system.reinitialize_energy_model(temperature=300., n_workers=1)

        # read coordinates
        traj = mdtraj.load_hdf5(molecule_path + "/traj.h5")
        coordinates = traj.xyz
        assert coordinates.shape[-2] == n_atoms, (f"pdb file ({n_atoms}) atoms does not match the "
                                                  f"data ({coordinates.shape[-2]}).")
    # determine n_dims
    experiment_params['network_params']['n_dims'] = len(coordinates[0].flat)

    # load model class and corresponding param class
    ModelClass = get_network_by_name(experiment_params['network_name'])
    ParamClass = BaseHParams
    hparams = ParamClass(**experiment_params['network_params'])

    # prepare the dataloaders
    train_split = experiment_params['training_params']['train_split']
    train_data, val_data, test_data = dataset_setter(
        coordinates, system, val_split=(.8 - train_split), test_split=.2, seed=42
    )
    print(f"{len(train_data)} training data, {len(val_data)} validation data, {len(test_data)} test data.")
    # create model
    if ModelClass.needs_energy_function:
        model = ModelClass(
            hparams, energy_model.energy, train_data=train_data, val_data=val_data
        )
    else:
        model = ModelClass(
            hparams, train_data=train_data, val_data=val_data
        )
    # load model state from previous experiment
    load_from_checkpoint = experiment_params.get('load_from_checkpoint', None)
    if load_from_checkpoint is not None:
        print(f"loading state_dict from data/lightning_logs/{load_from_checkpoint}/checkpoints/last.ckpt")
        checkpoint = torch.load(experiment_data_path + f"/lightning_logs/{load_from_checkpoint}/checkpoints/last.ckpt")
        model.load_state_dict(checkpoint['state_dict'])
    if "OppA" in experiment_param_name:  # TODO: this is ugly. find smth better
        i = experiment_param_name.rfind("/")
        experiment_param_name = experiment_param_name[i+1:]

    # fit the model
    model.fit(
        trainer_kwargs=experiment_params['trainer_kwargs'],
        logger_kwargs=dict(save_dir=lightning_logs, name=experiment_param_name)
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
    print("done")
