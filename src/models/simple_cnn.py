"""Simple CNN model used for MNIST baseline experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small two-convolution-layer CNN for 28x28 greyscale images."""

    def __init__(self, num_classes: int = 10):  # 10 classes for MNIST digits 0 to 9
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # preserve 28 x 28 image size
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # image is 7 x 7 after two pooling layers
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 x 28 -> 14 x 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 x 14 -> 7 x 7
        x = torch.flatten(x, 1)  # flatten all features except the batch dimension
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw class scores before softmax