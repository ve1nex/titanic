from pathlib import Path
import copy

import optuna
from omegaconf import OmegaConf

from train import run_fold
from utils import set_seed


def _set_path(config, path, value):
    parts = str(path).split(".")
    node = config
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


def _suggest(trial, name, spec):
    kind = str(spec.type)
    if kind == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high), step=int(spec.get("step", 1)))
    if kind == "float":
        return trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))
    raise ValueError(f"Unknown tuning type: {kind}")


def apply_best_params(config, best_params):
    for name, value in best_params.items():
        _set_path(config, config.tuning.search_space[name].path, value)


def run_tuning(config, features, labels):
    folds = [int(x) for x in config.tuning.folds_to_use]

    def objective(trial):
        trial_config = copy.deepcopy(config)
        for name, spec in config.tuning.search_space.items():
            _set_path(trial_config, spec.path, _suggest(trial, str(name), spec))
        scores = []
        for step, fold in enumerate(folds):
            set_seed(int(config.general.seed) + fold, bool(config.reproducibility.deterministic))
            result = run_fold(trial_config, features, labels, fold, checkpoint_root=Path(config.paths.path_to_tuning) / "tmp", trial=trial if bool(config.tuning.pruning.enabled) else None, trial_step_offset=step * int(config.training.num_epochs), save_artifacts=False)
            scores.append(result["score"])
        return float(np.mean(scores))

    import numpy as np
    db = Path(config.paths.path_to_optuna_db).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=int(config.general.seed), n_startup_trials=int(config.tuning.pruning.n_startup_trials)) if str(config.tuning.sampler) == "tpe" else optuna.samplers.RandomSampler(seed=int(config.general.seed))
    study = optuna.create_study(
        study_name=str(config.tuning.study_name),
        direction=str(config.metric.direction),
        sampler=sampler,
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=int(config.tuning.pruning.n_startup_trials), n_warmup_steps=int(config.tuning.pruning.n_warmup_steps)) if bool(config.tuning.pruning.enabled) else optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=int(config.tuning.n_trials))

    out = Path(config.paths.path_to_tuning)
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    OmegaConf.save(OmegaConf.create({"best_value": float(study.best_value), "best_params": dict(study.best_params)}), out / "best_params.yaml")
    return dict(study.best_params), study
