"""
SENetBiLSTM — Squeeze-and-Excitation ResNet + Bidirectional LSTM.

Based on:
  - SENet  : Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
  - ResNet : He et al., "Deep Residual Learning", CVPR 2016.
  - BiLSTM : Schuster & Paliwal, 1997.

Checkpoint : models/senet_bilstm_v1.0.pth
"""

import torch
import torch.nn as nn

from .blocks import SEResBlock1D


class SENetBiLSTM(nn.Module):
    """
    SE-ResNet + Bidirectional LSTM hybrid.

    SE attention blocks dynamically re-weight feature map channels before
    feeding the compressed sequence into a bidirectional LSTM.

    Architecture
    ------------
    Entry Conv (in → 64) → MaxPool (50 → 25 frames)
    → SEResBlock×4 (64→128→256[s=2,25→13]→256→512)
    → Permute → BiLSTM (hidden=128, 2-layer) → Global Avg Pool
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
        self.entry_conv = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.entry_bn   = nn.BatchNorm1d(64)
        self.relu       = nn.ReLU()
        self.pool       = nn.MaxPool1d(2)
        self.res_block1 = SEResBlock1D(64, 128)
        self.res_block2 = SEResBlock1D(128, 256, stride=2)
        self.res_block3 = SEResBlock1D(256, 256)
        self.res_block4 = SEResBlock1D(256, 512)
        self.bilstm     = nn.LSTM(512, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout    = nn.Dropout(0.5)
        self.fc_out     = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.entry_bn(self.entry_conv(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        return self.fc_out(self.dropout(torch.mean(lstm_out, dim=1)))
