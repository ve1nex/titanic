from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_training_curves(history, output_path):
    """Plot saved training loss, validation loss, metric, and learning rate."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training losses")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path.with_name(path.stem + "_loss.png"), dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["metric"], label="Validation metric")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation metric")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path.with_name(path.stem + "_metric.png"), dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], label="Learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning rate")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path.with_name(path.stem + "_lr.png"), dpi=140)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, output_path):
    """Save a confusion matrix from validation predictions."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix)
    fig, ax = plt.subplots(figsize=(7, 7))
    display.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
