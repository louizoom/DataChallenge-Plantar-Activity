"""
ResNetBiLSTM — Residual 1D CNN + Bidirectional LSTM.

Based on:
  - ResNet  : He et al., "Deep Residual Learning", CVPR 2016.
  - BiLSTM  : Schuster & Paliwal, "Bidirectional Recurrent Neural Networks", 1997.

Checkpoint : models/resnet_bilstm_v1.0.pth
"""

import torch
import torch.nn as nn

from .blocks import ResBlock1D


class ResNetBiLSTM(nn.Module):
    """
    Residual 1D CNN + Bidirectional LSTM hybrid.

    Architecture
    ------------
    Entry Conv (in → 32 ch) → MaxPool (50 → 25 frames)
    → ResBlock (32 → 64) → ResBlock (64 → 128)
    → BiLSTM (2 × 128) → Global Average Pooling
    → Dropout(0.5) → FC (256 → num_classes)

    Parameters
    ----------
    num_features : int
        Number of input sensor channels.
    num_classes : int
        Number of target action classes.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.entry_conv = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.entry_bn   = nn.BatchNorm1d(32)
        self.relu       = nn.ReLU()
        self.pool       = nn.MaxPool1d(2)
        self.res_block1 = ResBlock1D(32, 64)
        self.res_block2 = ResBlock1D(64, 128)
        self.bilstm     = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1        = nn.Linear(128 * 2, 64)
        self.dropout    = nn.Dropout(0.5)
        self.fc2        = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.entry_bn(self.entry_conv(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        return self.fc2(self.dropout(self.relu(self.fc1(avg_pool))))
