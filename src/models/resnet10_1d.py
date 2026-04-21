"""
ResNet10_1D — 10-layer Deep Residual Network for 1-D time-series (Champion).

Based on ResNet (He et al., "Deep Residual Learning", CVPR 2016),
adapted to 1-D sensor data. Achieved 78.23 % validation accuracy on the
plantar activity classification task.

Checkpoint : models/resnet10_1d_v1.0.pth
"""

import torch
import torch.nn as nn

from .blocks import ResBlock1D


class ResNet10_1D(nn.Module):
    """
    10-layer Deep Residual Network for 1-D time-series classification.

    Layer breakdown
    ---------------
    1   : Entry Conv1d (in → 64 ch), BN, ReLU, MaxPool   [50 → 25 frames]
    2–3 : ResBlock 1 — (64 → 128)                          [25 frames]
    4–5 : ResBlock 2 — (128 → 256, stride=2)               [13 frames]
    6–7 : ResBlock 3 — (256 → 256)                         [13 frames]
    8–9 : ResBlock 4 — (256 → 512)                         [13 frames]
    10  : Global Average Pooling + Dropout(0.4) + Linear (512 → num_classes)

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

        self.res_block1 = ResBlock1D(64, 128)
        self.res_block2 = ResBlock1D(128, 256, stride=2)
        self.res_block3 = ResBlock1D(256, 256)
        self.res_block4 = ResBlock1D(256, 512)

        self.dropout = nn.Dropout(0.4)
        self.fc_out  = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                                    # (B, F, T)
        x = self.pool(self.relu(self.entry_bn(self.entry_conv(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = torch.mean(x, dim=2)                                   # Global Avg Pool
        return self.fc_out(self.dropout(x))
