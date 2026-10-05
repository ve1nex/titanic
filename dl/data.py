from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from generating_dataset import load_raw_data, prepare_fold


def _load_npy(path):
    """Read a numeric array without permitting arbitrary pickle data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path, allow_pickle=False)


def load_training_data(config):
    """Load raw features for fold-local learned preprocessing."""
    features, labels, _ = load_raw_data(config, train=True)
    return features, labels


def load_test_data(config):
    """Load the original test array used by archived checkpoints."""
    return np.asarray(
        _load_npy(config.paths.path_to_test_features), dtype=str(config.data.dtype)
    )


class TrainDataset(Dataset):
    """Expose transformed tabular features and classification labels to PyTorch."""

    def __init__(self, features, labels, config):
        self.features = features
        self.labels = labels
        self.flatten = bool(config.data.flatten_for_mlp)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        if self.flatten:
            x = x.flatten()
        y = torch.as_tensor(self.labels[index], dtype=torch.long)
        return {"features": x, "labels": y}


class InferenceDataset(Dataset):
    """Expose transformed test features in their original row order."""

    def __init__(self, features, config):
        self.features = features
        self.flatten = bool(config.data.flatten_for_mlp)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        return x.flatten() if self.flatten else x


def _loader_kwargs(config, train):
    """Build training or ordered inference DataLoader options."""
    return {
        "batch_size": int(config.dataloader_params.batch_size),
        "shuffle": bool(config.dataloader_params.shuffle) if train else False,
        "num_workers": int(config.dataloader_params.num_workers),
        "pin_memory": bool(config.dataloader_params.pin_memory),
    }


def get_data_loader(features, labels, config, is_train):
    """Create a DataLoader for transformed training or validation samples."""
    return DataLoader(
        TrainDataset(features, labels, config), **_loader_kwargs(config, is_train)
    )


def get_inference_loader(features, config):
    """Create an ordered DataLoader for test samples."""
    return DataLoader(
        InferenceDataset(features, config), **_loader_kwargs(config, False)
    )


def get_fold_indices(features, labels, config, fold):
    """Return training and validation row positions for one stratified fold."""
    cv = StratifiedKFold(
        n_splits=int(config.split.n_splits),
        shuffle=bool(config.split.shuffle),
        random_state=int(config.general.seed) if bool(config.split.shuffle) else None,
    )
    for current, (train_idx, val_idx) in enumerate(cv.split(features, labels)):
        if current == int(fold):
            return np.asarray(train_idx), np.asarray(val_idx)
    raise ValueError(f"Fold {fold} does not exist")


def get_fold_loaders(features, labels, config, fold, preprocessor_path=None):
    """Fit a training-fold transform and save it with that fold's weights."""
    train_idx, val_idx = get_fold_indices(features, labels, config, fold)
    x_train, x_valid, preprocessor = prepare_fold(features, train_idx, val_idx)
    config.model.input_shape = [int(x_train.shape[1])]
    if preprocessor_path is not None:
        Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
    return (
        get_data_loader(x_train, labels[train_idx], config, True),
        get_data_loader(x_valid, labels[val_idx], config, False),
        train_idx,
        val_idx,
    )
