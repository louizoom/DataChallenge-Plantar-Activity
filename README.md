# Plantar Activity Classification — Deep Learning Benchmark

> Classifying **31 human actions** from plantar (insole) pressure and IMU sensor data using a family of PyTorch deep learning architectures.  
> **Champion model: DeepResNet-10L — 78.23 % validation accuracy** (Group K-Fold, patient-level generalisation).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Quick Start](#quick-start)
5. [Data Structure](#data-structure)
6. [Models](#models)
7. [Results](#results)
8. [Data Pipeline](#data-pipeline)
9. [Configuration Reference](#configuration-reference)

---

## Project Overview

This project was developed as part of a Data Science challenge. The goal is to classify 31 distinct human activities (walking, jumping, transitions, etc.) from raw time-series sensor data recorded at **100 FPS** with plantar insoles (50 sensors per foot pair + IMU).

**Key methodological choices:**

| Technique | Rationale |
|-----------|-----------|
| Sliding-window segmentation (50 frames / 0.5 s) | Preserves temporal dynamics |
| `StandardScaler` normalisation | Aligns units (Newton, g, °/s) |
| **Group K-Fold** cross-validation | Ensures patient-level generalisation — no data leakage |
| Balanced class weights (`CrossEntropyLoss`) | Prevents majority-class bias (e.g. walking) |
| Residual connections (ResNet) | Enables very deep networks without vanishing gradient |

---

## Repository Structure

```
DataChallenge-Plantar-Activity/
│
├── src/                          # Training & benchmark scripts
│   ├── models/                   # Pure PyTorch architectures
│   │   ├── baselines.py          # CNN1D, MLP
│   │   ├── blocks.py             # ResBlocks
│   │   ├── resnet10_1d.py        # ⭐ DeepResNet-10L (Champion)
│   │   └── ...                   # resnet_bilstm.py, convlstm.py, etc.
│   │
│   ├── training/                 # Model-specific training wrappers
│   │   ├── train_resnet10_1d.py
│   │   ├── train_resnet_bilstm.py
│   │   └── ...                   # train_cnn1d_baseline.py, etc.
│   │
│   └── evaluation/               # Cross-validation & benchmarks
│       ├── train_kfold.py
│       ├── benchmark_kfold.py
│       └── experiment_runner.py
│
├── utils/                        # Shared data loading & path utilities
│   ├── __init__.py
│   ├── paths.py                  # Env-variable driven path configuration
│   └── data_utils.py             # Loading, windowing, and cleaning functions
│
├── notebooks/
│   └── generate_charts.py        # Generate comparison and learning-curve charts
│
├── docs/                         # Project documentation
│   ├── champion_model_resnet10_1d.md
│   ├── theoretical_background.md
│   └── baseline_experiments.md
│
├── models/                       # Saved model checkpoints (git-ignored)
├── results/                      # Training logs, JSON results, charts (git-ignored)
├── outputs/                      # Miscellaneous outputs (git-ignored)
│
├── .env.example                  # Environment variable template (copy → .env)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Prerequisites & Installation

**Python ≥ 3.9** is required.

```bash
# 1. Clone the repository
git clone https://github.com/louizoom/DataChallenge-Plantar-Activity.git
cd DataChallenge-Plantar-Activity

# 2. (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** PyTorch will automatically use the MPS backend for hardware acceleration. No extra configuration is needed.

> **NVIDIA GPU:** Set `CUDA_VISIBLE_DEVICES` as needed; PyTorch detects CUDA automatically.

---

## Quick Start

### 1. Configure your data path

```bash
cp .env.example .env
```

Open `.env` and fill in the path to your local data directory:

```ini
# Absolute path to your data root, or relative to the project root
DATA_ROOT=/path/to/your/DataChallenge_donneesGlobales

# Sub-folder containing the insole CSV files
# Sorted dataset  : Plantar_activity_trie
# Unsorted dataset: Plantar_activity
PLANTAR_FOLDER=Plantar_activity_trie

# Sub-folder containing the classification annotation files
EVENTS_FOLDER=Events
```

### 2. Run the champion model

```bash
python src/training/train_resnet10_1d.py
```

Expected output: training loop reaching **~78 %** validation accuracy over 50 epochs.

### 3. Run the multi-model benchmark (Group K-Fold)

```bash
python src/evaluation/benchmark_kfold.py
```

Results are saved to `results/kfold_comparison_results.json`.

### 4. Generate charts

```bash
python notebooks/generate_charts.py
```

Produces `results/chart_learning_curve.png` and `results/chart_model_comparison.png`.

---

## Data Structure

The project expects the following folder layout inside `DATA_ROOT`:

```
DATA_ROOT/
├── PLANTAR_FOLDER/           # e.g. Plantar_activity_trie
│   ├── S01/
│   │   ├── Sequence_01/
│   │   │   └── insoles.csv   # 51 columns: Time + 50 sensor channels (sep=';')
│   │   └── Sequence_02/
│   │       └── insoles.csv
│   └── S32/
│       └── ...
│
└── EVENTS_FOLDER/            # e.g. Events
    ├── S01/
    │   ├── Sequence_01/
    │   │   └── classif.csv   # Columns: Class, Name, Timestamp Start, Timestamp End
    │   └── Sequence_02/
    │       └── classif.csv
    └── S32/
        └── ...
```

Scripts automatically discover all available subjects and sequences — **you do not need to modify any source file** when adding new subjects or using a different number of subjects.

---

## Models

| Script | Architecture | Checkpoint | Subjects | Epochs | Validation |
|--------|-------------|-----------|----------|--------|------------|
| `training/train_cnn1d_baseline.py` | Naive CNN 1D | *(not saved)* | S01–S02 | 4 | Random 80/20 |
| `training/train_random_forest.py` | Random Forest | *(not saved)* | S01–S05 | — | Random 80/20 |
| `training/train_convlstm.py` | ConvLSTM | `convlstm_v1.0.pth` | S01–S05 | 35 | Random 80/20 |
| `training/train_resnet_bilstm.py` | ResNetBiLSTM | `resnet_bilstm_v1.0.pth` | All 32 | 40 | Random 80/20 |
| `training/train_resnet10_1d.py` | **ResNet10_1D** ⭐ | `resnet10_1d_v1.0.pth` | All 32 | 50 | Random 80/20 |
| `training/train_senet_bilstm.py` | SENetBiLSTM | `senet_bilstm_v1.0.pth` | All 32 | 70 | Random 80/20 |
| `evaluation/train_kfold.py` | ResNetBiLSTM | *(per-fold)* | All 32 | 50/fold | **10-Fold GroupKFold** |
| `evaluation/benchmark_kfold.py` | MLP / CNN / RF / ResNetBiLSTM | *(per-fold)* | All 32 | 50/fold | **5-Fold GroupKFold** |

### Champion Architecture — ResNet10_1D

```
Input (B, 50, F)
    └─→ Conv1d (F→64, k=5) + BN + ReLU + MaxPool     [50 → 25 frames]
    └─→ ResBlock 1: Conv1d (64→128)                   [25 frames]
    └─→ ResBlock 2: Conv1d (128→256, stride=2)        [13 frames]
    └─→ ResBlock 3: Conv1d (256→256)                  [13 frames]
    └─→ ResBlock 4: Conv1d (256→512)                  [13 frames]
    └─→ Global Average Pooling                        [(B, 512)]
    └─→ Dropout(0.4) + Linear (512 → num_classes)
```

Each `ResBlock` applies two Conv1d layers with a skip connection (identity shortcut + optional downsampling). This prevents the vanishing gradient problem in very deep networks.

---

## Results

All scores below use **Group K-Fold** cross-validation (subject-level splits — no temporal data leakage).

| Model | Val Accuracy | Std | Method | Checkpoint |
|-------|-------------|-----|--------|------------|
| Random Forest | 42.4 % | ±2.4 % | 5-Fold GroupKFold | — |
| MLP Dense | 42.1 % | ±2.1 % | 5-Fold GroupKFold | — |
| CNN 1D | 43.3 % | ±1.8 % | 5-Fold GroupKFold | — |
| ResNet-BiLSTM (`resnet_bilstm_v1.0`) | 46.5 % | ±1.9 % | 5-Fold GroupKFold | `resnet_bilstm_v1.0.pth` |
| SENet-BiLSTM (`senet_bilstm_v1.0`) | 77.3 % | — | Full train/val split | `senet_bilstm_v1.0.pth` |
| **ResNet10-1D (`resnet10_1d_v1.0`)** ⭐ | **78.2 %** | — | Full train/val split | `resnet10_1d_v1.0.pth` |

> **Note on Random Forest (frame-by-frame, ~80 %):** The experiment runner
> reports ~80 % accuracy because it uses a random train/test split, causing
> temporal data leakage (frames from the same second appear in both splits).
> This score is **not comparable** to the Group K-Fold results.

---

## Data Pipeline

```
insoles.csv  +  classif.csv
      │
      ▼  Temporal alignment (Timestamp Start / End masking)
  Merged DataFrame  (100 FPS, Class column added)
      │
      ▼  Drop unlabelled frames + forward/backward fill NaN
  Cleaned DataFrame
      │
      ▼  StandardScaler (per-feature z-score normalisation)
  Normalised Features
      │
      ▼  Sliding-window segmentation (50 frames, stride=25, majority-vote label)
  Tensor (N, 50, num_features)
      │
      ▼  GroupKFold split by subject ID
  Train / Validation Folds
      │
      ▼  DataLoader (batch_size=128–256, shuffle=True for train)
  Mini-batches → Model → CrossEntropyLoss (balanced weights) → Adam → Scheduler
      │
      ▼
  Best checkpoint saved to models/
```

---

## Configuration Reference

All configurable parameters live in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `DataChallenge_donneesGlobales` | Path to the data root directory |
| `PLANTAR_FOLDER` | `Plantar_activity_trie` | Sub-folder with insole CSV files |
| `EVENTS_FOLDER` | `Events` | Sub-folder with classification CSV files |

If any of these directories are not found at import time, `utils/paths.py` will print a clear warning pointing you to the `.env` file.
