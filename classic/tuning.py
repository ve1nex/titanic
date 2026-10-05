from pathlib import Path
import copy

import optuna
from omegaconf import OmegaConf

from train import evaluate_cv_score


def _suggest(trial, name, spec):
    """Sample one integer, float, or categorical hyperparameter."""
    kind = str(spec.type)
    if kind == "int":
        return trial.suggest_int(
            name, int(spec.low), int(spec.high), step=int(spec.get("step", 1))
        )
    if kind == "float":
        return trial.suggest_float(
            name, float(spec.low), float(spec.high), log=bool(spec.get("log", False))
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))
    raise ValueError(f"Unknown tuning type: {kind}")


def apply_best_params(config, best_params):
    """Apply recorded Optuna parameters to the model configuration."""
    name = str(config.model.name)
    for key, value in best_params.items():
        config.models[name][key] = value


def run_tuning(config, X, y):
    """Evaluate the configured hyperparameter search with cross-validation."""
    name = str(config.model.name)
    if name not in config.tuning.search_spaces:
        raise ValueError(f"No search space for {name}")
    space = config.tuning.search_spaces[name]

    def objective(trial):
        trial_cfg = copy.deepcopy(config)
        for key, spec in space.items():
            trial_cfg.models[name][key] = _suggest(trial, str(key), spec)
        return evaluate_cv_score(
            trial_cfg, X, y, folds_to_use=config.tuning.folds_to_use
        )

    db = Path(config.paths.path_to_optuna_db).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    sampler = (
        optuna.samplers.TPESampler(seed=int(config.general.seed))
        if str(config.tuning.sampler) == "tpe"
        else optuna.samplers.RandomSampler(seed=int(config.general.seed))
    )
    study = optuna.create_study(
        study_name=str(config.tuning.study_name),
        direction=str(config.metric.direction),
        sampler=sampler,
        storage=f"sqlite:///{db}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=int(config.tuning.n_trials))

    out = Path(config.paths.path_to_tuning)
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    OmegaConf.save(
        OmegaConf.create(
            {
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
            }
        ),
        out / "best_params.yaml",
    )
    return dict(study.best_params), study
