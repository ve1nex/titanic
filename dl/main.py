from config import config
from data import load_training_data
from predict import inference
from train import train
from tuning import apply_best_params, run_tuning
from utils import prepare_experiment, save_config_snapshot, save_dataset_metadata, set_seed


def fit(config):
    set_seed(int(config.general.seed), bool(config.reproducibility.deterministic))
    features, labels = load_training_data(config)
    prepare_experiment(config)
    save_config_snapshot(config)
    save_dataset_metadata(config, features, labels)

    if bool(config.tuning.enabled):
        best_params, study = run_tuning(config, features, labels)
        apply_best_params(config, best_params)
        save_config_snapshot(config)
        if bool(config.logging.prints):
            print(f"Best CV: {study.best_value:.6f}")
            print(f"Best params: {best_params}")

    train(config, features, labels)


def main():
    if str(config.general.mode) == "train":
        fit(config)
    elif str(config.general.mode) == "inference":
        inference(config)
    else:
        raise ValueError("general.mode must be 'train' or 'inference'")


if __name__ == "__main__":
    main()
