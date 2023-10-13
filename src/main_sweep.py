import os
import sys

import yaml

from main import run_experiment

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

    # find the parameter to sweep over in params
    # sweeping over multible params would require either extracting all combinations or preset combinations
    s = params['to_sweep']
    start = 0
    # find seperators (demarked with ":")
    stop = s.find(":")
    subparams = params
    while stop != -1:
        stop = s[start:].find(":")
        # if this is the final key don't execute the last part of the loop because
        # we want to access the last nested stage (the actual parameter) always with
        # dict[key] so the operation still mutates the dicts
        if stop == -1:
            break
        key = s[start:][:stop]
        subparams = subparams[key]
        start += stop + 1
    key = s[start:]
    # save the param list to loop over because it will get overwritten now
    sweeping_params = subparams[key]
    # loop over param values and run main for each of them
    for p in sweeping_params:
        subparams[key] = p
        # (ab)use that params is mutable and got mutated via subparams above
        run_experiment(params, param_name, data_path)
