from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent

config = {
    # --- General ---
    "general": {
        "experiment_name": "KNNbest",
        "seed": 0xFACED,
        "mode": "inference",             # train / inference
        "overwrite_experiment": False,
    },

    # --- Paths ---
    "paths": {
        "path_to_train_dataset": str(PROJECT_ROOT.parent / "train.csv"),
        "path_to_test_dataset": str(PROJECT_ROOT.parent / "test.csv"),
        "path_to_checkpoints": str(PROJECT_ROOT / "checkpoints" / "${general.experiment_name}"),
        "path_to_model": "${paths.path_to_checkpoints}/model.joblib",
        "path_to_oof": "${paths.path_to_checkpoints}/oof_predictions.csv",
        "path_to_predictions": "${paths.path_to_checkpoints}/predictions.csv",
        "path_to_plots": "${paths.path_to_checkpoints}/plots",
        "path_to_config_snapshot": "${paths.path_to_checkpoints}/config.yaml",
        "path_to_metadata": "${paths.path_to_checkpoints}/metadata.json",
        "path_to_optuna_db": str(PROJECT_ROOT / "optuna" / "${general.experiment_name}.db"),
        "path_to_tuning": "${paths.path_to_checkpoints}/tuning",
    },

    # --- Data ---
    "data": {
        "target": "Survived",
        "id_column": "PassengerId",
        "drop_columns": ["Ticket", "Cabin"],
    },

    # --- Preprocessing ---
    "preprocessing": {
        "numeric_imputer": "mean",
        "categorical_imputer": "most_frequent",
        "scale_numeric": True,
        "encode_categorical": True,
    },

    # --- Titanic feature engineering ---
    "feature_engineering": {
        "enabled": True,                  # Title from Name
    },

    # --- Validation ---
    "split": {
        "n_splits": 5,
        "folds_to_train": [0, 1, 2, 3, 4],
        "shuffle": True,
    },

    # --- Training ---
    "training": {
        "save_oof_predictions": True,
        "train_final_model": True,
    },

    # --- Model ---
    "model": {
        "name": "KNeighborsClassifier",
    },

    # Only models used in the final Titanic workflow are kept.
    "models": {
        "LogisticRegression": {
            "C": float("inf"),            # A1: effectively no L2 regularization
            "solver": "lbfgs",
            "max_iter": 5000,
        },
        "KNeighborsClassifier": {
            "n_neighbors": 8,
            "weights": "uniform",
            "p": 1,
            "metric": "minkowski",
            "n_jobs": -1,
        },
        "RandomForestClassifier": {
            "n_estimators": 350,
            "max_depth": 6,
            "min_samples_split": 18,
            "min_samples_leaf": 4,
            "max_features": None,
            "random_state": "${general.seed}",
            "n_jobs": -1,
        },
        "XGBClassifier": {
            "n_estimators": 300,
            "learning_rate": 0.051492960003745855,
            "min_child_weight": 5.22226831701564,
            "gamma": 0.0840195392056468,
            "max_depth": 6,
            "subsample": 0.8685332897567278,
            "colsample_bytree": 0.609062303945643,
            "reg_alpha": 0.057225176664358425,
            "reg_lambda": 1.7084420539981688,
            "random_state": "${general.seed}",
            "n_jobs": -1,
        },
    },

    # --- Metric ---
    "metric": {
        "name": "accuracy_score",
        "direction": "maximize",
    },

    # --- Optuna ---
    # Kept because it was used for RF/XGBoost tuning.
    "tuning": {
        "enabled": False,
        "n_trials": 40,
        "sampler": "tpe",                # tpe / random
        "study_name": "${general.experiment_name}_tuning",
        "folds_to_use": "${split.folds_to_train}",
        "search_spaces": {
            "LogisticRegression": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "RandomForestClassifier": {
                "n_estimators": {"type": "int", "low": 200, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 4, "high": 8},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            },
            "XGBClassifier": {
                "n_estimators": {"type": "int", "low": 280, "high": 320, "step": 10},
                "learning_rate": {"type": "float", "low": 0.04, "high": 0.1, "log": True},
                "max_depth": {"type": "int", "low": 5, "high": 7},
                "min_child_weight": {"type": "float", "low": 4.0, "high": 8.0, "log": True},
                "gamma": {"type": "float", "low": 0.01, "high": 0.5},
                "subsample": {"type": "float", "low": 0.6, "high": 0.9},
                "colsample_bytree": {"type": "float", "low": 0.4, "high": 0.8},
                "reg_alpha": {"type": "float", "low": 0.03, "high": 0.08, "log": True},
                "reg_lambda": {"type": "float", "low": 1.0, "high": 2.0, "log": True},
            },
            "KNeighborsClassifier": {
                "n_neighbors": {
                    "type": "int",
                    "low": 3,
                    "high": 20,
                    "step": 1,
                },
                "weights": {
                    "type": "categorical",
                    "choices": ["uniform", "distance"],
                },
                "p": {
                    "type": "categorical",
                    "choices": [1, 2],
                },
            },
        },
    },

    # --- Output ---
    "visualization": {
        "save_validation_plot": True,
        "save_feature_importance": True,
    },
    "logging": {
        "prints": True,
    },
}

config = OmegaConf.create(config)
