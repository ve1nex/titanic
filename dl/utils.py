import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def set_seed(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(config):
    requested = str(config.training.device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def ensure_directories(config):
    Path(config.paths.path_to_checkpoints).mkdir(parents=True, exist_ok=True)
    Path(config.paths.path_to_fold_checkpoints).mkdir(parents=True, exist_ok=True)
    Path(config.paths.path_to_plots).mkdir(parents=True, exist_ok=True)


def prepare_experiment(config):
    root = Path(config.paths.path_to_checkpoints)
    if root.exists() and str(config.general.mode) == "train":
        if bool(config.general.overwrite_experiment):
            shutil.rmtree(root)
        else:
            raise FileExistsError(f"Experiment already exists: {root}")
    ensure_directories(config)


def save_config_snapshot(config):
    OmegaConf.save(config, config.paths.path_to_config_snapshot)


def save_dataset_metadata(config, features, labels):
    payload = {
        "features_shape": list(features.shape),
        "features_dtype": str(features.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
    }
    Path(config.paths.path_to_metadata).write_text(json.dumps(payload, indent=2), encoding="utf-8")
