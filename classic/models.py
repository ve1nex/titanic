from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

MODELS = {
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}


def get_model(config):
    name = str(config.model.name)
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS)}")
    params = OmegaConf.to_container(config.models[name], resolve=True)
    return MODELS[name](**params)
