"""
model.py
--------
SimpleCNN model definition for MNIST digit classification.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple 2-layer CNN for MNIST digit classification.

    Architecture:
        Conv2d(1, 32) -> ReLU -> MaxPool
        Conv2d(32, 64) -> ReLU -> MaxPool
        Flatten
        Linear(3136, 128) -> ReLU -> Dropout(0.5)
        Linear(128, 10)

    Input shape : (N, 1, 28, 28)
    Output shape: (N, 10)  -- raw logits for 10 digit classes
    """

    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (N, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (N, 64, 14, 14)
        self.pool  = nn.MaxPool2d(2)                               # halves H & W

        # Fully-connected layers
        self.fc1     = nn.Linear(64 * 7 * 7, 128)  # 64*7*7 = 3136
        self.fc2     = nn.Linear(128, 10)           # 10 digit classes
        self.dropout = nn.Dropout(p=0.5)            # regularisation

    def forward(self, x):
        # Conv block 1: (N,1,28,28) -> (N,32,14,14)
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2: (N,32,14,14) -> (N,64,7,7)
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten: (N,64,7,7) -> (N,3136)
        x = x.view(x.size(0), -1)
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
