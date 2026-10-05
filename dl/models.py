import math
import torch.nn as nn


def _activation(name):
    """Resolve the configured PyTorch activation class."""
    if not hasattr(nn, name):
        raise ValueError(f"Unknown activation: {name}")
    return getattr(nn, name)


class MLPBackbone(nn.Module):
    """Build the tested dense layers with optional BatchNorm and dropout."""

    def __init__(
        self, input_shape, hidden_dims, dropout=0.0, activation="ReLU", batchnorm=False
    ):
        super().__init__()
        activation_cls = _activation(activation)
        layers = [nn.Flatten()]
        current = int(math.prod(input_shape))
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.append(nn.Linear(current, hidden))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation_cls())
            layers.append(nn.Dropout(float(dropout)))
            current = hidden
        self.network = nn.Sequential(*layers)
        self.feature_dim = current

    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    """Map learned features to the output class scores."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(int(input_dim), int(output_dim)))

    def forward(self, x):
        return self.network(x)


class UniversalDLModel(nn.Module):
    """Connect the existing backbone and prediction head interfaces."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = PredictionHead(backbone.feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def get_model(config):
    """Instantiate the model and parameters selected in config."""
    if str(config.model.name) != "MLP":
        raise ValueError("Titanic DL supports MLP")
    p = config.model.params
    backbone = MLPBackbone(
        input_shape=list(config.model.input_shape),
        hidden_dims=list(p.hidden_dims),
        dropout=float(p.dropout),
        activation=str(p.activation),
        batchnorm=bool(p.batchnorm),
    )
    return UniversalDLModel(backbone, int(config.general.num_classes))


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
