import importlib.util
import os
import sys
import warnings

default_rel_params_path = "../../params/"
default_rel_logs_path = "../../logs/"

param_path = sys.argv[1]
if not os.path.isabs(param_path):
    param_path = default_rel_params_path + param_path
spec = importlib.util.spec_from_file_location("params", param_path)
params = importlib.util.module_from_spec(spec)
sys.modules["params"] = params
spec.loader.exec_module(params)



def main():
    model = params.Experiment(hparams=params.hparams)
    trainer_kwargs = getattr(params, "trainer_kwargs")
    logger_kwargs = getattr(params, "logger_kwargs")
    if logger_kwargs is not None:
        save_dir = logger_kwargs.get("save_dir", "")
        if not os.path.isabs(save_dir):
            save_dir = default_rel_logs_path + save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger_kwargs["save_dir"] = save_dir
    else:
        warnings.warn("No logger_kwargs provided. Logging to default location.")
    model.fit(trainer_kwargs=trainer_kwargs, logger_kwargs=logger_kwargs)


if __name__ == "__main__":
    main()
