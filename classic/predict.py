from pathlib import Path

import joblib
import pandas as pd

from data import load_csv, prepare_dataframe
from features import feature_engineering


def inference(config):
    model_path = Path(config.paths.path_to_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    pipeline = joblib.load(model_path)
    df = feature_engineering(prepare_dataframe(load_csv(config.paths.path_to_test_dataset), config), config)

    ids = df[str(config.data.id_column)].copy()
    X_test = df.drop(columns=[str(config.data.id_column)], errors="ignore")

    predictions = pipeline.predict(X_test)
    #probabilities = pipeline.predict_proba(X_test)[:, 1]

    output = pd.DataFrame({
        str(config.data.id_column): ids.to_numpy(),
        str(config.data.target): predictions,
        #"Probability": probabilities,
    })
    path = Path(config.paths.path_to_predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    if bool(config.logging.prints):
        print(f"Saved predictions: {path}")
    return output
