import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from checkpointing import load_checkpoint, save_checkpoint
from data import get_fold_loaders
from losses import get_loss
from metrics import get_metric, is_improvement, outputs_to_predictions
from models import count_parameters, get_model
from optimizers import get_optimizer
from schedulers import get_scheduler, step_scheduler
from utils import resolve_device, set_seed
from visualization import save_confusion_matrix, save_training_curves


def train_one_epoch(config, model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Train", leave=False) if bool(config.logging.prints) else loader:
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def validate_one_epoch(config, model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    outputs, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Valid", leave=False) if bool(config.logging.prints) else loader:
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            out = model(x)
            total += float(loss_fn(out, y).detach().cpu())
            outputs.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    return total / max(1, len(loader)), get_metric(config, targets, outputs), outputs, targets


def run_fold(config, features, labels, fold, checkpoint_root=None, trial=None, trial_step_offset=0, save_artifacts=True):
    set_seed(int(config.general.seed) + int(fold), bool(config.reproducibility.deterministic))
    device = resolve_device(config)
    root = Path(checkpoint_root or config.paths.path_to_fold_checkpoints)
    fold_dir = root / f"fold_{fold}"
    if save_artifacts:
        fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, val_idx = get_fold_loaders(features, labels, config, fold)
    model = get_model(config).to(device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    loss_fn = get_loss(config)
    total_params, trainable_params = count_parameters(model)
    if bool(config.logging.prints):
        print(f"Device: {device} | Parameters: {total_params:,} ({trainable_params:,} trainable)")

    best_metric = None
    best_outputs = None
    best_targets = None
    best_path = fold_dir / "best.pt"
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "metric": [], "lr": []}

    for epoch in range(int(config.training.num_epochs)):
        started = time.time()
        train_loss = train_one_epoch(config, model, train_loader, optimizer, loss_fn, device)
        val_loss, metric, outputs, targets = validate_one_epoch(config, model, val_loader, loss_fn, device)
        if scheduler is not None:
            step_scheduler(scheduler, val_loss)

        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["metric"].append(metric)
        history["lr"].append(lr)

        if is_improvement(config, metric, best_metric):
            best_metric = metric
            best_outputs = outputs
            best_targets = targets
            epochs_without_improvement = 0
            if save_artifacts and bool(config.training.save_best):
                save_checkpoint(best_path, model, optimizer, scheduler, None, epoch + 1, metric, best_metric, 0)
        else:
            epochs_without_improvement += 1

        if trial is not None:
            trial.report(metric, step=int(trial_step_offset) + epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if bool(config.logging.prints):
            print(
                f"Fold {fold} | Epoch {epoch + 1}/{config.training.num_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | metric={metric:.4f} | "
                f"best={best_metric:.4f} | lr={lr:.2e} | {int(time.time()-started)}s"
            )
        if epochs_without_improvement >= int(config.training.early_stopping_epochs):
            if bool(config.logging.prints):
                print("Early stopping.")
            break

    if save_artifacts:
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if bool(config.visualization.save_training_curves):
            save_training_curves(history, Path(config.paths.path_to_plots) / f"fold_{fold}_training")
        if bool(config.visualization.save_confusion_matrix):
            save_confusion_matrix(best_targets, outputs_to_predictions(best_outputs), Path(config.paths.path_to_plots) / f"fold_{fold}_confusion_matrix.png")

    return {"fold": int(fold), "score": float(best_metric), "outputs": best_outputs, "targets": best_targets, "val_idx": val_idx}


def train(config, features, labels):
    folds = [int(x) for x in config.split.folds_to_train]
    oof_outputs = np.zeros((len(labels), int(config.general.num_classes)), dtype=np.float32)
    oof_labels = np.asarray(labels).copy()
    seen = np.zeros(len(labels), dtype=bool)
    scores = []

    for fold in folds:
        result = run_fold(config, features, labels, fold)
        oof_outputs[result["val_idx"]] = result["outputs"]
        seen[result["val_idx"]] = True
        scores.append(result["score"])

    np.savez(config.paths.path_to_oof, outputs=oof_outputs[seen], labels=oof_labels[seen])
    mean, std = float(np.mean(scores)), float(np.std(scores))
    if bool(config.logging.prints):
        print(f"\nCV: {mean:.4f} ± {std:.4f}")
    return mean, std
