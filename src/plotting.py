from lightning_bg.utils import load_data, dataset_setter, load_model_kwargs
from lightning_bg.models import get_network_by_name
import torch
import numpy as np
import os

def load_from_checkpoint(checkpoint_path, data_path, model_class):
    coordinates, system = load_data(data_path)
    train_split = .7
    train_data, val_data, test_data = dataset_setter(coordinates, system, val_split=(.8 - train_split),
                                                     test_split=.2, seed=42)
    ModelClass = get_network_by_name(model_class)
    ParamClass = ModelClass.hparams_type
    hparams = torch.load(checkpoint_path.rstrip("/") + "/checkpoints/last.ckpt")['hyper_parameters']
    model_kwargs = load_model_kwargs(ModelClass, train_data, val_data, system)
    model = ModelClass.load_from_checkpoint(checkpoint_path.rstrip("/") + "/checkpoints/last.ckpt",
                                            hparams=hparams,
                                            **model_kwargs
                                            )
    return model, test_data, system

print(os.getcwd())
model_1b4z_RNVPICfwkl, test_1b4z, system_1b4z = load_from_checkpoint("data/lightning_logs/OppA/Peptides/1b4z/RNVPICfwkl/version_8", "data/Molecules/OppA/Peptides/1b4z", "RNVPICfwkl")
model_1b4z_RNVPfwkl, _, _ = load_from_checkpoint("data/lightning_logs/OppA/Peptides/1b4z/RNVPfwkl/version_8", "data/Molecules/OppA/Peptides/1b4z", "RNVPfwkl")

energyIC = system_1b4z.energy_model.energy(model_1b4z_RNVPICfwkl.sample((10_000,))).cpu().detach().numpy()
# energyIC2 = system_1b4z.energy_model.energy(model_1b4z_RNVPICfwkl2.sample((10_000,))).cpu().detach().numpy()
energyCC = system_1b4z.energy_model.energy(model_1b4z_RNVPfwkl.sample((10_000,))).cpu().detach().numpy()
energy_target = system_1b4z.energy_model.energy(test_1b4z[:10_000]).cpu().detach().numpy()

# save energyIC as csv
np.savetxt("plots/1b4z/energyIC_version8.csv", energyIC, delimiter=",")
np.savetxt("plots/1b4z/energyCC_version8.csv", energyCC, delimiter=",")
np.savetxt("plots/1b4z/energy_target.csv", energy_target, delimiter=",")