import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from data import build_preprocessor
from models import get_model
from utils import get_metric
from visualization import save_feature_importance, save_validation_plot


def get_cv(config):
    """Create the seeded validation splitter used by this project."""
    return StratifiedKFold(
        n_splits=int(config.split.n_splits),
        shuffle=bool(config.split.shuffle),
        random_state=int(config.general.seed) if bool(config.split.shuffle) else None,
    )


def build_training_pipeline(X, config):
    """Fit learned preprocessing together with the selected estimator."""
    return Pipeline(
        [
            ("preprocessing", build_preprocessor(X, config)),
            ("model", get_model(config)),
        ]
    )


def evaluate_cv_score(config, X, y, folds_to_use=None):
    """Evaluate selected CV folds without saving model artifacts."""
    requested = set(int(x) for x in (folds_to_use or config.split.folds_to_train))
    scores = []
    for fold, (tr, va) in enumerate(get_cv(config).split(X, y)):
        if fold not in requested:
            continue
        model = build_training_pipeline(X.iloc[tr], config)
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(get_metric(config, y.iloc[va], model.predict(X.iloc[va])))
    if not scores:
        raise ValueError("No folds were evaluated")
    return float(np.mean(scores))


def train(config, X, y):
    """Train validation folds and persist OOF predictions and CV results."""
    requested = set(int(x) for x in config.split.folds_to_train)
    if not requested or not requested.issubset(set(range(int(config.split.n_splits)))):
        raise ValueError("Select valid validation folds")
    scores = []
    oof_pred = np.zeros(len(y), dtype=np.asarray(y).dtype)
    fold_ids = np.full(len(y), -1, dtype=int)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    mask = np.zeros(len(y), dtype=bool)
    start = time.time()

    for fold, (tr, va) in enumerate(get_cv(config).split(X, y)):
        if fold not in requested:
            continue
        model = build_training_pipeline(X.iloc[tr], config)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        score = get_metric(config, y.iloc[va], pred)
        scores.append(score)
        oof_pred[va] = pred
        oof_prob[va] = prob
        mask[va] = True
        fold_ids[va] = fold
        if bool(config.logging.prints):
            print(f"Fold {fold} | {config.metric.name}: {score:.4f}")

    if bool(config.training.save_oof_predictions):
        pd.DataFrame(
            {
                "row_index": np.arange(len(y))[mask],
                "target": np.asarray(y)[mask],
                "prediction": oof_pred[mask],
                "probability": oof_prob[mask],
                "fold": fold_ids[mask],
            }
        ).to_csv(config.paths.path_to_oof, index=False)

    save_validation_plot(np.asarray(y)[mask], oof_pred[mask], config)

    final_model = None
    if bool(config.training.train_final_model):
        final_model = build_training_pipeline(X, config)
        final_model.fit(X, y)
        joblib.dump(final_model, config.paths.path_to_model)
        save_feature_importance(final_model, X, config)

    mean, std = float(np.mean(scores)), float(np.std(scores))
    from utils import save_cv_metadata

    save_cv_metadata(config, scores, fold_ids, mask)
    if bool(config.logging.prints):
        print(f"\nCV {config.metric.name}: {mean:.4f} ± {std:.4f}")
        print(f"Time: {int(time.time() - start)} s")
    return final_model, scores
