import numpy as np
import sklearn.metrics


def outputs_to_predictions(outputs):
    return np.argmax(np.asarray(outputs), axis=1)


def get_metric(config, y_true, outputs):
    name = str(config.metric.name)
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown metric: {name}")
    return float(getattr(sklearn.metrics, name)(y_true, outputs_to_predictions(outputs), **dict(config.metric.params)))


def is_improvement(config, current, best):
    if best is None:
        return True
    return current > best if str(config.metric.direction) == "maximize" else current < best
