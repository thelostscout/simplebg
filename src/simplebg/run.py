import importlib.util
import os
import sys
import warnings

default_rel_params_path = "../../params/"
default_rel_logs_path = "../../logs/"


def main():
    # load hparams from provided file path
    param_path = sys.argv[1]
    # if no absolute path is provided, assume it is relative to the default params path and hence add the default
    # param path in front
    if not os.path.isabs(param_path):
        param_path = default_rel_params_path + param_path
    # importlib magic
    spec = importlib.util.spec_from_file_location("params", param_path)
    params = importlib.util.module_from_spec(spec)
    # here we define the file we loaded as "params" module
    sys.modules["params"] = params
    spec.loader.exec_module(params)
    # load the model class from the params module and instantiate it with the provided hparams
    model = params.Experiment(hparams=params.hparams)
    # need trainer_kwargs and logger_kwargs from params as well
    trainer_kwargs = getattr(params, "trainer_kwargs")
    logger_kwargs = getattr(params, "logger_kwargs")
    # quality of life: if no absolute logger path is provided, assume it is relative to the default logs path and
    # hence add the default param path in front
    if logger_kwargs is not None:
        save_dir = logger_kwargs.get("save_dir", "")
        if not os.path.isabs(save_dir):
            save_dir = default_rel_logs_path + save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger_kwargs["save_dir"] = save_dir
    else:
        warnings.warn("No logger_kwargs provided. Logging to default location.")
    # fit the model
    model.fit(trainer_kwargs=trainer_kwargs, logger_kwargs=logger_kwargs)


if __name__ == "__main__":
    main()
