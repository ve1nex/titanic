from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_csv(path):
    """Read a CSV file and reject a missing dataset path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def prepare_dataframe(df, config):
    """Remove configured columns and apply fixed missing-value rules."""
    existing = [c for c in config.data.drop_columns if c in df.columns]
    return df.drop(columns=existing).copy()


def split_features_target(df, config):
    """Exclude target and ID columns from the model features."""
    target = str(config.data.target)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing")
    drop = [target]
    if config.data.id_column and config.data.id_column in df.columns:
        drop.append(str(config.data.id_column))
    return df.drop(columns=drop), df[target]


def build_preprocessor(X, config):
    """Build numeric imputation/scaling and categorical encoding steps."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    num_steps = [
        ("imputer", SimpleImputer(strategy=str(config.preprocessing.numeric_imputer)))
    ]
    if bool(config.preprocessing.scale_numeric):
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        (
            "imputer",
            SimpleImputer(strategy=str(config.preprocessing.categorical_imputer)),
        )
    ]
    if bool(config.preprocessing.encode_categorical):
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    return ColumnTransformer(transformers, remainder="drop")
