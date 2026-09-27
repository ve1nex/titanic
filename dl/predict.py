from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from checkpointing import load_checkpoint
from data import get_inference_loader, load_test_data
from models import get_model
from utils import resolve_device


def _predict_one(config, checkpoint_path, loader, device):
    model = get_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    batches = []
    with torch.no_grad():
        for x in tqdm(loader, desc="Inference", leave=False) if bool(config.logging.prints) else loader:
            logits = model(x.to(device))
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def predict_features(config, features, folds=None):
    loader = get_inference_loader(features, config)
    device = resolve_device(config)
    folds = [int(x) for x in (folds if folds is not None else config.split.folds_to_inference)]
    root = Path(config.paths.path_to_fold_checkpoints)
    predictions = []
    for fold in folds:
        path = root / f"fold_{fold}" / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        predictions.append(_predict_one(config, path, loader, device))
    return np.mean(predictions, axis=0)


def inference(config):
    features = load_test_data(config)
    probabilities = predict_features(config, features)
    predictions = np.argmax(probabilities, axis=1)

    pred_path = Path(config.paths.path_to_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_path, predictions)
    np.save(pred_path.with_name("probabilities.npy"), probabilities[:, 1])

    if bool(config.logging.prints):
        print(f"Predictions saved to: {pred_path}")
        print(f"Probabilities saved to: {pred_path.with_name('probabilities.npy')}")
    return predictions
