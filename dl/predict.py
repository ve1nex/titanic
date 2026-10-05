import copy
from pathlib import Path

import joblib
import numpy as np
import torch

from data import get_inference_loader, load_test_data
from generating_dataset import load_raw_data
from models import get_model
from utils import resolve_device


def predict_features(config, features, folds=None):
    """Average fold probabilities using each fold's saved transform."""
    device = resolve_device(config)
    folds = [
        int(f)
        for f in (folds if folds is not None else config.split.folds_to_inference)
    ]
    if not folds:
        raise ValueError("Select at least one inference fold")
    root = Path(config.paths.path_to_fold_checkpoints)
    predictions = []
    for fold in folds:
        directory = root / f"fold_{fold}"
        checkpoint = torch.load(
            directory / "best.pt", map_location=device, weights_only=True
        )
        fold_config = copy.deepcopy(config)
        if checkpoint.get("input_shape") is not None:
            fold_config.model.input_shape = checkpoint["input_shape"]
        if checkpoint.get("preprocessing") == "per_fold":
            matrix = joblib.load(directory / "preprocessor.joblib").transform(features)
        else:
            # Archived weights use the unchanged arrays supplied with the project.
            matrix = load_test_data(config)
        model = get_model(fold_config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        batches = []
        with torch.no_grad():
            for x in get_inference_loader(
                np.asarray(matrix, dtype=np.float32), fold_config
            ):
                batches.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        predictions.append(np.concatenate(batches))
    return np.mean(predictions, axis=0)


def inference(config):
    """Save the fold-averaged labels and positive-class probabilities."""
    features, _, ids = load_raw_data(config, train=False)
    probabilities = predict_features(config, features)
    if len(probabilities) != len(ids):
        raise ValueError("Prediction count differs from test passengers")
    path = Path(config.paths.path_to_predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.argmax(probabilities, axis=1)
    np.save(path, predictions)
    np.save(path.with_name("probabilities.npy"), probabilities[:, 1])
    if config.logging.prints:
        print(f"Saved predictions: {path}")
    return predictions
