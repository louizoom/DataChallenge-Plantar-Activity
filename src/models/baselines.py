"""
Baseline and simple architectures used for quick comparisons.

Classes
-------
MLP_Simple      : Fully-connected dense network baseline.
CNN1D_Simple    : Two-block 1D CNN baseline.
"""

import torch
import torch.nn as nn


class CNN1D_Simple(nn.Module):
    """
    Two-block 1D CNN baseline.
    """
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 12, 128)  # Assuming window size 50 -> pool to 25 -> pool to 12
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc2(self.relu(self.fc1(x)))


class MLP_Simple(nn.Module):
    """
    Fully-connected dense network baseline.
    """
    def __init__(self, num_features: int, num_classes: int, window_size: int = 50) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features * window_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(self.flatten(x))))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class CNN1D_Dynamic(nn.Module):
    """
    A 1D CNN baseline that computes the flattened dimension dynamically.
    Works for any window size (e.g. 20, 60 as used in the experiment runner).
    """
    def __init__(self, num_features: int, num_classes: int, window_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Compute flattened size dynamically
        dummy = torch.zeros(1, num_features, window_size)
        out = self.pool(self.relu(self.conv1(dummy)))
        out = self.pool(self.relu(self.conv2(out)))
        flat_dim = out.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flat_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
