import json
import random
import shutil
from pathlib import Path

import numpy as np
import sklearn.metrics
from omegaconf import OmegaConf


def set_seed(seed):
    """Set Python, NumPy, and, when used, PyTorch random seeds."""
    random.seed(int(seed))
    np.random.seed(int(seed))


def ensure_directories(config):
    """Create the directories required for local experiment artifacts."""
    Path(config.paths.path_to_checkpoints).mkdir(parents=True, exist_ok=True)
    Path(config.paths.path_to_plots).mkdir(parents=True, exist_ok=True)


def prepare_experiment(config):
    """Prepare output directories while protecting existing experiments."""
    root = Path(config.paths.path_to_checkpoints)
    if (
        root.exists()
        and bool(config.general.overwrite_experiment)
        and str(config.general.mode) == "train"
    ):
        shutil.rmtree(root)
    elif (
        root.exists()
        and str(config.general.mode) == "train"
        and not bool(config.general.overwrite_experiment)
    ):
        raise FileExistsError(f"Experiment already exists: {root}")
    ensure_directories(config)


def save_config_snapshot(config):
    """Save the run configuration alongside its experiment artifacts."""
    OmegaConf.save(config, config.paths.path_to_config_snapshot)


def save_dataset_metadata(config, df):
    """Record the data schema used by the current experiment."""
    payload = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    Path(config.paths.path_to_metadata).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def get_metric(config, y_true, y_pred):
    """Compute the configured validation metric from model predictions."""
    name = str(config.metric.name)
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown metric: {name}")
    return float(getattr(sklearn.metrics, name)(y_true, y_pred))


def save_cv_metadata(config, scores, fold_ids, mask):
    """Append CV metrics without replacing the saved dataset schema."""
    path = Path(config.paths.path_to_metadata)
    payload = json.loads(path.read_text()) if path.exists() else {}
    folds = sorted(set(int(f) for f in fold_ids[mask]))
    payload.update(
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores)),
        fold_scores=dict(zip(map(str, folds), map(float, scores))),
        oof_complete=bool(mask.all()),
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
