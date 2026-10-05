"""Titanic DL table preprocessing fitted separately inside every validation fold."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(config, train=True):
    """Load the seven original MLP features and preserve raw passenger order."""
    path = (
        config.paths.path_to_train_dataset
        if train
        else config.paths.path_to_test_dataset
    )
    df = pd.read_csv(path)
    labels = df["Survived"].to_numpy(dtype=np.int64) if train else None
    ids = df["PassengerId"].to_numpy()
    features = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    return features, labels, ids


def build_preprocessor():
    """Build the median/scaling/one-hot transform used by the Titanic MLP."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["Pclass", "Age", "SibSp", "Parch", "Fare"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                ["Sex", "Embarked"],
            ),
        ]
    )


def prepare_fold(features, train_idx, val_idx):
    """Learn statistics on training passengers and reuse them on validation passengers."""
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(features.iloc[train_idx])
    x_valid = preprocessor.transform(features.iloc[val_idx])
    return (
        np.asarray(x_train, dtype=np.float32),
        np.asarray(x_valid, dtype=np.float32),
        preprocessor,
    )
