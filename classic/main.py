from checks import check_data_leakage
from config import config
from data import load_csv, prepare_dataframe, split_features_target
from features import feature_engineering
from predict import inference
from train import train
from tuning import apply_best_params, run_tuning
from utils import prepare_experiment, save_config_snapshot, save_dataset_metadata, set_seed


def fit(config):
    set_seed(config.general.seed)
    df = load_csv(config.paths.path_to_train_dataset)
    prepare_experiment(config)
    df = feature_engineering(prepare_dataframe(df, config), config)
    save_dataset_metadata(config, df)
    save_config_snapshot(config)
    X, y = split_features_target(df, config)
    check_data_leakage(X, y, config)

    if bool(config.tuning.enabled):
        best_params, study = run_tuning(config, X, y)
        apply_best_params(config, best_params)
        save_config_snapshot(config)
        if bool(config.logging.prints):
            print(f"Best CV: {study.best_value:.6f}")
            print(f"Best params: {best_params}")

    train(config, X, y)


def main():
    if str(config.general.mode) == "train":
        fit(config)
    elif str(config.general.mode) == "inference":
        inference(config)
    else:
        raise ValueError("general.mode must be 'train' or 'inference'")


if __name__ == "__main__":
    main()
