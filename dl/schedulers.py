import torch
from omegaconf import OmegaConf


def get_scheduler(config, optimizer):
    """Instantiate the scheduler when its feature flag is enabled."""
    if not bool(config.scheduler.enabled):
        return None
    name = str(config.scheduler.name)
    if not hasattr(torch.optim.lr_scheduler, name):
        raise ValueError(f"Unknown scheduler: {name}")
    params = OmegaConf.to_container(config.scheduler.params, resolve=True)
    return getattr(torch.optim.lr_scheduler, name)(optimizer, **params)


def step_scheduler(scheduler, val_loss):
    """Advance the scheduler using validation loss when required."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
