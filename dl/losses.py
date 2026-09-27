import torch.nn as nn
from omegaconf import OmegaConf


def get_loss(config):
    name = str(config.loss.name)
    if not hasattr(nn, name):
        raise ValueError(f"Unknown torch loss: {name}")
    params = OmegaConf.to_container(config.loss.params, resolve=True)
    return getattr(nn, name)(**params)
