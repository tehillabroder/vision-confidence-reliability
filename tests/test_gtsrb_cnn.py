"""Tests for the GTSRB baseline CNN."""

import torch
from src.models.gtsrb_cnn import GTSRBCNN
def test_gtsrb_cnn_returns_one_score_per_class():
    # confirm the model returns scores for all 43 classes
    model = GTSRBCNN()
    images = torch.randn(4, 3, 64, 64)

    outputs = model(images)

    assert outputs.shape == (4, 43)

def test_gtsrb_cnn_supports_configured_class_count():
    # check that the final layer follows the supplied class count
    model = GTSRBCNN(num_classes=5)
    images = torch.randn(2, 3, 64, 64)

    outputs = model(images)

    assert outputs.shape == (2, 5)