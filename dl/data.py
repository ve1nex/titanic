from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


def _load_npy(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path, allow_pickle=False)


def load_training_data(config):
    features = np.asarray(_load_npy(config.paths.path_to_train_features), dtype=str(config.data.dtype))
    labels = np.asarray(_load_npy(config.paths.path_to_train_labels), dtype=np.int64)
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same length")
    return features, labels


def load_test_data(config):
    return np.asarray(_load_npy(config.paths.path_to_test_features), dtype=str(config.data.dtype))


class TrainDataset(Dataset):
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
    def __init__(self, features, config):
        self.features = features
        self.flatten = bool(config.data.flatten_for_mlp)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        return x.flatten() if self.flatten else x


def _loader_kwargs(config, train):
    return {
        "batch_size": int(config.dataloader_params.batch_size),
        "shuffle": bool(config.dataloader_params.shuffle) if train else False,
        "num_workers": int(config.dataloader_params.num_workers),
        "pin_memory": bool(config.dataloader_params.pin_memory),
    }


def get_data_loader(features, labels, config, is_train):
    return DataLoader(TrainDataset(features, labels, config), **_loader_kwargs(config, is_train))


def get_inference_loader(features, config):
    return DataLoader(InferenceDataset(features, config), **_loader_kwargs(config, False))


def get_fold_indices(features, labels, config, fold):
    cv = StratifiedKFold(
        n_splits=int(config.split.n_splits),
        shuffle=bool(config.split.shuffle),
        random_state=int(config.general.seed) if bool(config.split.shuffle) else None,
    )
    for current, (train_idx, val_idx) in enumerate(cv.split(features, labels)):
        if current == int(fold):
            return np.asarray(train_idx), np.asarray(val_idx)
    raise ValueError(f"Fold {fold} does not exist")


def get_fold_loaders(features, labels, config, fold):
    train_idx, val_idx = get_fold_indices(features, labels, config, fold)
    return (
        get_data_loader(features[train_idx], labels[train_idx], config, True),
        get_data_loader(features[val_idx], labels[val_idx], config, False),
        train_idx,
        val_idx,
    )
