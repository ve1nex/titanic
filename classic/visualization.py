from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_validation_plot(y_true, y_pred, config):
    """Save validation diagnostics for the current prediction task."""
    if not bool(config.visualization.save_validation_plot):
        return
    Path(config.paths.path_to_plots).mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("OOF confusion matrix")
    plt.tight_layout()
    plt.savefig(
        Path(config.paths.path_to_plots) / "confusion_matrix.png",
        dpi=140,
        bbox_inches="tight",
    )
    plt.close()


def save_feature_importance(pipeline, X, config):
    """Plot available model feature importances or linear coefficients."""
    if not bool(config.visualization.save_feature_importance):
        return
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]
    names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importance = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        return
    top_n = min(25, len(names))
    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(9, max(5, top_n * 0.28)))
    plt.barh(np.asarray(names)[idx], importance[idx])
    plt.xlabel("Importance")
    plt.title(f"Feature importance: {config.model.name}")
    plt.tight_layout()
    plt.savefig(
        Path(config.paths.path_to_plots) / "feature_importance.png",
        dpi=140,
        bbox_inches="tight",
    )
    plt.close()
