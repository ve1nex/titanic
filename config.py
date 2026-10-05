"""Titanic entry-point and final voting settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
config = {
    "mode": "report",
    "pipeline": "final",
    "output_dir": str(ROOT / "outputs"),
    "classic_folder": "classic",
    "dl_folder": "dl",
    "new_experiment_prefix": "clean_cv_v2",
    "ensemble": {
        "rf_experiment": "clean_cv_v1_rf",
        "xgb_experiment": "clean_cv_v1_xgb",
        "dl_experiment": "clean_cv_v1_dl",
        "voting": "soft",
        "weights": [1, 1, 1],
        "threshold": 0.5,
    },
}
