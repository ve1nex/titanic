"""One entry point for Titanic saved results, training, and saved-model inference."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import ROOT, config
from ensemble.ensemble_soft import run_ensemble


def _fold_ids(y):
    """Recover the original five-fold assignment used by the saved final models."""
    folds = np.full(len(y), -1, dtype=int)
    for fold, (_, va) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0xFACED).split(
            np.zeros(len(y)), y
        )
    ):
        folds[va] = fold
    return folds


def _cv(y, prediction, folds):
    """Calculate saved-prediction accuracy without fitting any model."""
    values = [
        float(np.mean(y[folds == f] == prediction[folds == f]))
        for f in np.unique(folds)
    ]
    return float(np.mean(values)), float(np.std(values))


def report(cfg, ensemble_cfg=None):
    """Build the configured final vote and verify its OOF metric from saved predictions."""
    settings = cfg["ensemble"] if ensemble_cfg is None else ensemble_cfg
    y = pd.read_csv(ROOT / "train.csv").Survived.to_numpy()
    default_folds = _fold_ids(y)
    rows, classic_oof = [], {}
    for path in sorted((ROOT / "classic/checkpoints").glob("*/oof_predictions.csv")):
        frame = pd.read_csv(path).sort_values("row_index")
        if not np.array_equal(frame.row_index, np.arange(len(y))) or not np.array_equal(
            frame.target, y
        ):
            raise ValueError(f"OOF passenger order/targets differ: {path}")
        folds = frame.fold.to_numpy() if "fold" in frame else default_folds
        mean, std = _cv(y, frame.prediction.to_numpy(), folds)
        rows.append(
            {
                "experiment": path.parent.name,
                "cv_mean": mean,
                "cv_std": std,
                "metric": "accuracy",
            }
        )
        classic_oof[path.parent.name] = frame
    dl_root = ROOT / "dl/checkpoints" / settings["dl_experiment"]
    with np.load(dl_root / "oof_predictions.npz", allow_pickle=False) as data:
        logits, labels = data["outputs"], data["labels"]
        if not np.array_equal(labels, y):
            raise ValueError("MLP OOF targets differ from raw passenger order")
        if "row_index" in data and not np.array_equal(
            data["row_index"], np.arange(len(y))
        ):
            raise ValueError("MLP OOF is incomplete or shuffled")
        dl_folds = data["fold"] if "fold" in data else default_folds
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    mlp = (exponentials / exponentials.sum(axis=1, keepdims=True))[:, 1]
    mean, std = _cv(y, (mlp >= 0.5).astype(int), dl_folds)
    rows.append(
        {
            "experiment": settings["dl_experiment"],
            "cv_mean": mean,
            "cv_std": std,
            "metric": "accuracy",
        }
    )
    frames = [
        classic_oof[settings["rf_experiment"]],
        classic_oof[settings["xgb_experiment"]],
    ]
    for frame in frames:
        if "fold" in frame and not np.array_equal(frame.fold, dl_folds):
            raise ValueError("Classic and DL OOF validation folds differ")
    values = np.column_stack([frames[0].probability, frames[1].probability, mlp])
    if settings["voting"] == "hard":
        values = np.column_stack(
            [frames[0].prediction, frames[1].prediction, (mlp >= 0.5).astype(int)]
        )
    prediction = (
        np.average(values, weights=settings["weights"], axis=1) >= settings["threshold"]
    ).astype(int)
    mean, std = _cv(y, prediction, dl_folds)
    rows.append(
        {
            "experiment": "final_ensemble",
            "cv_mean": mean,
            "cv_std": std,
            "metric": "accuracy",
        }
    )
    output = Path(cfg["output_dir"])
    run_ensemble(ROOT, settings, output)
    results = pd.DataFrame(rows)
    results.to_csv(output / "results.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "cv_mean": mean,
                "cv_std": std,
                "global_oof_accuracy": float(np.mean(prediction == y)),
                "voting": settings["voting"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Final CV: {mean:.4f} ± {std:.4f}")
    print(f"Submission: {output / 'submission.csv'}")
    return results


def _run_pipeline(cfg, pipeline, mode, experiment, model=None):
    """Run the existing modules in separate processes to avoid config import collisions."""
    args = [
        sys.executable,
        str(ROOT / cfg[f"{pipeline}_folder"] / "main.py"),
        f"general.mode={mode}",
        f"general.experiment_name={experiment}",
    ]
    if model:
        args.append(f"model.name={model}")
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    """Dispatch a saved-results report or an explicitly requested training/inference run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["report", "train", "inference"], default=config["mode"]
    )
    parser.add_argument(
        "--pipeline", choices=["classic", "dl", "final"], default=config["pipeline"]
    )
    args = parser.parse_args()
    if args.mode == "report":
        report(config)
        return
    settings = copy.deepcopy(config["ensemble"])
    if args.mode == "train":
        prefix = config["new_experiment_prefix"]
        settings.update(
            rf_experiment=prefix + "_rf",
            xgb_experiment=prefix + "_xgb",
            dl_experiment=prefix + "_dl",
        )
    if args.pipeline in {"classic", "final"}:
        _run_pipeline(
            config,
            "classic",
            args.mode,
            settings["rf_experiment"],
            "RandomForestClassifier",
        )
        if args.pipeline == "final":
            _run_pipeline(
                config,
                "classic",
                args.mode,
                settings["xgb_experiment"],
                "XGBClassifier",
            )
    if args.pipeline in {"dl", "final"}:
        _run_pipeline(config, "dl", args.mode, settings["dl_experiment"])
    if args.pipeline == "final":
        report(config, settings)


if __name__ == "__main__":
    main()
