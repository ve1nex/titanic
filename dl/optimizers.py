import torch
from omegaconf import OmegaConf


def get_optimizer(config, model):
    name = str(config.optimizer.name)
    if not hasattr(torch.optim, name):
        raise ValueError(f"Unknown optimizer: {name}")
    params = OmegaConf.to_container(config.optimizer.params, resolve=True)
    return getattr(torch.optim, name)(model.parameters(), **params)
