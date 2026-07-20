"""Small CNN for GTSRB baseline experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
class GTSRBCNN(nn.Module):
    """A three-layer CNN for RGB traffic-sign images."""

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # force feature map spatial dimensions to (C, 4, 4) regardless of input resolution        
        self.feature_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(F.relu(self.conv1(images)))
        features = self.pool(F.relu(self.conv2(features)))
        features = self.pool(F.relu(self.conv3(features)))
        features = self.feature_pool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(F.relu(self.fc1(features)))
        return self.fc2(features)