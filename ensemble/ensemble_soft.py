"""ID-aligned Titanic probability averaging and hard voting from saved predictions."""

from pathlib import Path
import numpy as np
import pandas as pd


def run_ensemble(root, config, output_dir):
    """Combine RF, XGBoost, and MLP using the method recorded in config.py."""
    root, output_dir = Path(root), Path(output_dir)
    ids = pd.read_csv(root / "test.csv")["PassengerId"].to_numpy()
    probabilities, labels = [], []
    for experiment in [config["rf_experiment"], config["xgb_experiment"]]:
        path = root / "classic/checkpoints" / experiment / "predictions.csv"
        if not path.exists() or "Probability" not in pd.read_csv(path).columns:
            fallback = {"A3_rf": "a3_rf_proba.csv", "A4_xgb": "a4_xgb_proba.csv"}.get(
                experiment
            )
            if fallback is None:
                raise ValueError(f"No saved probabilities for {experiment}")
            path = root / "ensemble" / fallback
        frame = pd.read_csv(path)
        if frame.PassengerId.duplicated().any() or set(frame.PassengerId) != set(ids):
            raise ValueError(f"Passenger IDs differ for {experiment}")
        frame = frame.set_index("PassengerId").loc[ids]
        probabilities.append(frame.Probability.to_numpy())
        labels.append(frame.Survived.to_numpy())
    dl_root = root / "dl/checkpoints" / config["dl_experiment"]
    probabilities.append(np.load(dl_root / "probabilities.npy", allow_pickle=False))
    labels.append(np.load(dl_root / "predictions.npy", allow_pickle=False))
    if any(len(p) != len(ids) for p in probabilities + labels):
        raise ValueError("Saved prediction count differs from test passengers")
    matrix = np.column_stack(probabilities)
    if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    weights = np.asarray(config["weights"], dtype=float)
    if (
        len(weights) != 3
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError(
            "Use three finite non-negative voting weights with a positive sum"
        )
    if config["voting"] == "soft":
        values = np.average(matrix, weights=weights, axis=1)
    elif config["voting"] == "hard":
        values = np.average(np.column_stack(labels), weights=weights, axis=1)
    else:
        raise ValueError("voting must be soft or hard")
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "PassengerId": ids,
            "Survived": (values >= float(config["threshold"])).astype(int),
        }
    )
    submission.to_csv(output_dir / "submission.csv", index=False)
    return submission
