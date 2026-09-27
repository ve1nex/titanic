import json
import random
import shutil
from pathlib import Path

import numpy as np
import sklearn.metrics
from omegaconf import OmegaConf


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def ensure_directories(config):
    Path(config.paths.path_to_checkpoints).mkdir(parents=True, exist_ok=True)
    Path(config.paths.path_to_plots).mkdir(parents=True, exist_ok=True)


def prepare_experiment(config):
    root = Path(config.paths.path_to_checkpoints)
    if root.exists() and bool(config.general.overwrite_experiment) and str(config.general.mode) == "train":
        shutil.rmtree(root)
    elif root.exists() and str(config.general.mode) == "train" and not bool(config.general.overwrite_experiment):
        raise FileExistsError(f"Experiment already exists: {root}")
    ensure_directories(config)


def save_config_snapshot(config):
    OmegaConf.save(config, config.paths.path_to_config_snapshot)


def save_dataset_metadata(config, df):
    payload = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    Path(config.paths.path_to_metadata).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_metric(config, y_true, y_pred):
    name = str(config.metric.name)
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown metric: {name}")
    return float(getattr(sklearn.metrics, name)(y_true, y_pred))
