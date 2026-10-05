import numpy as np
import sklearn.metrics


def outputs_to_predictions(outputs):
    """Select class labels from raw model output scores."""
    return np.argmax(np.asarray(outputs), axis=1)


def get_metric(config, y_true, outputs):
    """Compute the configured validation metric from model predictions."""
    name = str(config.metric.name)
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown metric: {name}")
    return float(
        getattr(sklearn.metrics, name)(
            y_true, outputs_to_predictions(outputs), **dict(config.metric.params)
        )
    )


def is_improvement(config, current, best):
    """Compare a new score using the configured metric direction."""
    if best is None:
        return True
    return (
        current > best if str(config.metric.direction) == "maximize" else current < best
    )
