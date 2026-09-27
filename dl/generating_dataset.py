from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT.parent


def generate_dataset():
    train = pd.read_csv(DATASET_ROOT / "train.csv")
    test = pd.read_csv(DATASET_ROOT / "test.csv")

    # Target
    y_train = train["Survived"].to_numpy(dtype=np.int64)

    # Убираем target
    X_train = train.drop(columns=["Survived"])
    X_test = test.copy()

    # Пока не используем эти признаки
    drop_columns = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
    ]

    X_train = X_train.drop(columns=drop_columns)
    X_test = X_test.drop(columns=drop_columns)

    # Числовые признаки
    numeric_columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    # Категориальные признаки
    categorical_columns = [
        "Sex",
        "Embarked",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    # Fit только на train
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Сохраняем
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    np.save(data_dir / "train_features.npy", X_train)
    np.save(data_dir / "train_labels.npy", y_train)
    np.save(data_dir / "test_features.npy", X_test)

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print("input_shape:", X_train.shape[1])


if __name__ == "__main__":
    generate_dataset()
