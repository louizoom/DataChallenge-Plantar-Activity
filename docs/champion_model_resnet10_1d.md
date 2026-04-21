# 🏆 Technical Fact Sheet: Champion Model (ResNet10_1D)

This document details the characteristics of the champion model of the plantar sensor analytical study. It outperformed all other tested architectures.

## 1. Model Identity Card
*   **Architecture Name**: ResNet10_1D (10-Layer Deep Residual Network)
*   **Task**: Multi-class Classification (31 human actions)
*   **Overall Accuracy (Validation)**: **78.23%** ⭐
*   **Associated Code Files**: `src/training/train_resnet10_1d.py` (Training script) and `models/resnet10_1d_v1.0.pth` (Saved neural weights)

---

## 2. Neural Architecture (Why "10 Layers"?)
One of the greatest victories of modern Data Science is the "Residual" network (ResNet). Generally, stacking too many layers makes the machine forget the initial data (Vanishing Gradient). This model uses **Skip-Connections** (a bridge that bypasses the layer) to keep the memory of the original sole signal.

The model totals exactly **10 layers with learnable weights** (hence the "10_1D"):

1.  **Initial Extraction Layer (1 layer)**:
    *   `Conv1d` (64 filters, kernel size=5). Roughly processes the raw signal.
    *   Followed by a `MaxPool` to reduce the temporal size of the window from 50 frames to 25 frames.
2.  **Residual Block 1 (2 layers)**: Keeps the signal size at 25 frames, increases to 128 filters to find small patterns (e.g., heel strike).
3.  **Residual Block 2 (2 layers)**: Uses a `Stride=2` (slow sliding) to compress time by half (25 frames -> 13 frames), and doubles the conceptual depth to 256 filters.
4.  **Residual Block 3 (2 layers)**: Deep extraction at 256 filters over 13 temporal frames.
5.  **Residual Block 4 (2 layers)**: The final highly abstract extraction at 512 filters.
6.  **Final Extrapolation (1 layer)**:
    *   Uses `Global Average Pooling` to summarize the meaning of the 13 remaining "timesteps" into a single score of conceptual significance.
    *   The final layer: `Linear` (Dense) transforms the 512 abstract feature filters into **31 action probabilities**.

---

## 3. Parameters, Training, and Data

AI is only effective thanks to the foundations and hyperparameters used to constrain it:

*   **Data Format (Sliding Windows)**: Temporal windows of 50 indices (~0.8 sec of action per window), with a step of 25 indices to ensure continuity.
*   **Rebalancing (Class Imbalance)**: Over-represented classes (like Walking) were mathematically penalized by the `compute_class_weight` function. Conversely, if the network made a mistake on a rare action (e.g., a Jump), the error loss was multiplied by a large coefficient to force learning.
*   **Loss Function**: `CrossEntropyLoss` (ideal for exclusive multi-category classifications).
*   **Optimizer (The Engine)**: `Adam` combined with a dynamic `ReduceLROnPlateau` scheduler. (If the model was stuck for 4 epochs, the optimizer reduced the learning rate by 50%).
*   **Training Epochs**: **50 Epochs.**

---

## 4. Why did it WIN against the others?

In the Machine Learning world, you must find the perfect boundary ("*The Sweet Spot*") between a naive model and an overfitting model.

1.  **Against basic models (MLP at 42%, CNN1D at 43%)**: The ResNet10_1D triumphed because biomechanical data is too complex. Walking or stumbling requires abstraction that 3 layers physically cannot compute. The depth (10 layers) was absolutely required.
2.  **Against excessive models (SENetBiLSTM at 77.3%)**: When we added even more complexity to the deep network (Squeeze-and-Excitation Attention + BiLSTM temporal layers), **the model became overfitted**. It memorized the tics and "noise" of the patients' soles, becoming less generalized and less robust on unknown data.

**The ResNet10_1D is the perfect entropy balance for this project!**
