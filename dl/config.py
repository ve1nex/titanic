from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent

config = {
    # --- General ---
    "general": {
        "experiment_name": "dl_baseline_iter40",
        "seed": 0xFACED,
        "mode": "inference",             # train / inference
        "num_classes": 2,
        "overwrite_experiment": False,
    },

    # --- Paths ---
    "paths": {
        "path_to_train_features": str(PROJECT_ROOT / "data" / "train_features.npy"),
        "path_to_train_labels": str(PROJECT_ROOT / "data" / "train_labels.npy"),
        "path_to_test_features": str(PROJECT_ROOT / "data" / "test_features.npy"),
        "path_to_checkpoints": str(PROJECT_ROOT / "checkpoints" / "${general.experiment_name}"),
        "path_to_fold_checkpoints": "${paths.path_to_checkpoints}/folds",
        "path_to_oof": "${paths.path_to_checkpoints}/oof_predictions.npz",
        "path_to_predictions": "${paths.path_to_checkpoints}/predictions.npy",
        "path_to_plots": "${paths.path_to_checkpoints}/plots",
        "path_to_config_snapshot": "${paths.path_to_checkpoints}/config.yaml",
        "path_to_metadata": "${paths.path_to_checkpoints}/metadata.json",
        "path_to_optuna_db": str(PROJECT_ROOT / "optuna" / "${general.experiment_name}.db"),
        "path_to_tuning": "${paths.path_to_checkpoints}/tuning",
    },

    # --- Reproducibility ---
    "reproducibility": {
        "deterministic": True,
    },

    # --- Data ---
    "data": {
        "dtype": "float32",
        "flatten_for_mlp": True,
    },

    # --- Validation ---
    "split": {
        "n_splits": 5,
        "folds_to_train": [0, 1, 2, 3, 4],
        "folds_to_inference": [0, 1, 2, 3, 4],
        "shuffle": True,
    },

    # --- Training ---
    "training": {
        "num_epochs": 30,
        "early_stopping_epochs": 7,
        "device": "cpu",                 # cpu / cuda / mps / auto
        "save_best": True,
    },

    # --- Dataloader ---
    "dataloader_params": {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": True,
    },

    # --- Optuna ---
    # Kept because it was used for the MLP experiments.
    "tuning": {
        "enabled": False,
        "n_trials": 10,
        "sampler": "tpe",
        "study_name": "${general.experiment_name}_tuning",
        "folds_to_use": [0, 1, 2, 3, 4],
        "pruning": {
            "enabled": True,
            "n_startup_trials": 5,
            "n_warmup_steps": 3,
        },
        # Generic path-based search. Example:
        "search_space": {
            "dropout": {
                "path": "model.params.dropout",
                "type": "float",
                "low": 0.1,
                "high": 0.5,
            },
        },
    },

    # --- Model ---
    "model": {
        "name": "MLP",
        "input_shape": [10],
        "params": {
            "hidden_dims": [20, 28],
            "dropout": 0.0,
            "activation": "ReLU",
            "batchnorm": False,
        },
    },

    # --- Optimizer ---
    "optimizer": {
        "name": "AdamW",
        "params": {
            "lr": 0.011497532816911557,
            "weight_decay": 0.0761432174738362,
        },
    },

    # --- Scheduler ---
    "scheduler": {
        "enabled": True,
        "name": "ReduceLROnPlateau",
        "params": {
            "mode": "min",
            "factor": 0.7,
            "patience": 3,
            "min_lr": 1e-5,
        },
    },

    # --- Loss / metric ---
    "loss": {
        "name": "CrossEntropyLoss",
        "params": {},
    },
    "metric": {
        "name": "accuracy_score",
        "direction": "maximize",
        "params": {},
    },

    # --- Output ---
    "visualization": {
        "save_training_curves": True,
        "save_confusion_matrix": True,
    },
    "logging": {
        "prints": True,
    },
}

config = OmegaConf.create(config)
