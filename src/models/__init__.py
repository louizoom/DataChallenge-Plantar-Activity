"""
Model registry for the Plantar Activity Classification project.

All model architectures are exposed from this package so that training
and evaluation scripts only need a single import line:

    from src.models import ResNet10_1D, ResNetBiLSTM, SENetBiLSTM, ConvLSTM

Building blocks (ResBlock1D, SEBlock1D, SEResBlock1D) are also available
for custom architectures:

    from src.models.blocks import ResBlock1D
"""

from .baselines import CNN1D_Simple, MLP_Simple, CNN1D_Dynamic
from .blocks import ResBlock1D, SEBlock1D, SEResBlock1D
from .convlstm import ConvLSTM
from .resnet_bilstm import ResNetBiLSTM
from .resnet10_1d import ResNet10_1D
from .senet_bilstm import SENetBiLSTM

__all__ = [
    # Building blocks
    "ResBlock1D",
    "SEBlock1D",
    "SEResBlock1D",
    # Full architectures
    "ConvLSTM",
    "ResNetBiLSTM",
    "ResNet10_1D",
    "SENetBiLSTM",
    # Baselines
    "CNN1D_Simple",
    "MLP_Simple",
    "CNN1D_Dynamic",
]
