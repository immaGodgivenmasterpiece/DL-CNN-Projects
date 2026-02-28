import torch
import torch.nn as nn


class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (N, 3, 32, 32)  -> (N, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (N, 32, 16, 16) -> (N, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (N, 64, 8, 8)   -> (N, 128, 8, 8)
        # Pooling (shared across all conv blocks)
        self.pool = nn.MaxPool2d(2, 2)                             # halves H and W each time
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)                    # (N, 2048) -> (N, 256)
        self.fc2 = nn.Linear(256, 10)                              # (N, 256)  -> (N, 10)
        # Dropout (applied before last output, not after)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1: (N, 3, 32, 32) -> (N, 32, 16, 16)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Conv block 2: (N, 32, 16, 16) -> (N, 64, 8, 8)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # Conv block 3: (N, 64, 8, 8) -> (N, 128, 4, 4)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        # Flatten: (N, 128, 4, 4) -> (N, 2048)
        x = x.view(x.size(0), -1)
        # FC + Dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)  # raw logits (no softmax — CrossEntropyLoss handles it)
        return x


class CIFAR10_CNN_BN(nn.Module):
    """CIFAR-10 CNN with BatchNorm (v2)."""
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # (N, 3, 32, 32)  -> (N, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # (N, 32, 16, 16) -> (N, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (N, 64, 8, 8)   -> (N, 128, 8, 8)
        # BatchNorm — must match conv output channels
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)                    # (N, 2048) -> (N, 256)
        self.fc2 = nn.Linear(256, 10)                              # (N, 256)  -> (N, 10)
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))  # (N, 3, 32, 32)  -> (N, 32, 16, 16)
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))  # (N, 32, 16, 16) -> (N, 64, 8, 8)
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))  # (N, 64, 8, 8)   -> (N, 128, 4, 4)
        # Flatten
        x = x.view(x.size(0), -1)                                  # (N, 2048)
        # FC + Dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)  # raw logits
        return x


if __name__ == "__main__":
    for name, cls in [("CIFAR10_CNN", CIFAR10_CNN), ("CIFAR10_CNN_BN", CIFAR10_CNN_BN)]:
        model = cls()
        print(f"\n{'='*40}\n{name}\n{'='*40}")
        print(model)
        dummy = torch.randn(1, 3, 32, 32)
        print("Output shape:", model(dummy).shape)  # should be [1, 10]