"""
ConvLSTM — Two-block 1D CNN followed by a stacked LSTM.

Reference: Shi et al., "Convolutional LSTM Network", NeurIPS 2015.
Checkpoint : models/convlstm_v1.0.pth
"""

import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    """
    Two-block 1D CNN followed by a stacked LSTM for temporal classification.

    Convolutional blocks extract local spatial patterns from the sensor signal;
    the LSTM models long-range temporal dependencies across the compressed
    sequence.

    Parameters
    ----------
    num_features : int
        Number of input sensor channels.
    num_classes : int
        Number of target action classes.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)                # length / 2

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)                # length / 4

        self.lstm    = nn.LSTM(64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1     = nn.Linear(128, 64)
        self.relu3   = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                                  # (B, F, T)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)                                  # (B, T', F')
        lstm_out, _ = self.lstm(x)
        out = self.fc2(self.dropout(self.relu3(self.fc1(lstm_out[:, -1, :]))))
        return out
