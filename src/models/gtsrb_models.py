"""Build GTSRB classification models."""

from __future__ import annotations
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18
from src.models.gtsrb_cnn import GTSRBCNN

RESNET18_WEIGHTS = {"IMAGENET1K_V1": ResNet18_Weights.IMAGENET1K_V1}
MOBILENET_V2_WEIGHTS = {"IMAGENET1K_V1": MobileNet_V2_Weights.IMAGENET1K_V1}

def _resolve_pretrained_weights(model_name: str, pretrained_weights: str | None):
    if pretrained_weights is None:
        return None

    if model_name == "ResNet18":
        options = RESNET18_WEIGHTS
    elif model_name == "MobileNetV2":
        options = MOBILENET_V2_WEIGHTS
    else:
        raise ValueError(f"{model_name} does not support pretrained weights.")

    if pretrained_weights not in options:
        raise ValueError(f"Unsupported pretrained weights for {model_name}: {pretrained_weights}")

    return options[pretrained_weights]

def build_gtsrb_model(model_name: str, num_classes: int = 43, pretrained_weights: str | None = None) -> nn.Module:
    """Build one supported GTSRB classifier."""
    if model_name == "GTSRBCNN":
        if pretrained_weights is not None:
            raise ValueError("GTSRBCNN does not support pretrained weights.")
        return GTSRBCNN(num_classes=num_classes)

    if model_name == "ResNet18":
        model = resnet18(weights=_resolve_pretrained_weights(model_name, pretrained_weights))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "MobileNetV2":
        model = mobilenet_v2(weights=_resolve_pretrained_weights(model_name, pretrained_weights))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported GTSRB model: {model_name}")