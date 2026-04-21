"""
5-Fold Group K-Fold Benchmark — Multi-Architecture Comparison.

Compares four model families side by side using identical 5-fold Group K-Fold
cross-validation (no temporal data leakage, patient-level splits):

    • MLP_Simple    — fully-connected dense network
    • CNN1D_Simple  — two-block 1D CNN
    • ResNetBiLSTM   — residual CNN + bidirectional LSTM
    • Random Forest — mean-pooled frame features (scikit-learn)

Results are saved to ``results/kfold_comparison_results.json``.

Usage
-----
    python src/benchmark_kfold.py

Output
------
    results/kfold_comparison_results.json — mean accuracy and std per model.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.dirname(_here)
_project_root = os.path.dirname(_src_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _src_root)

from utils import load_and_merge_data, create_windows_with_ids, PLANTAR_DIR, RESULTS_METRICS_DIR
from models import ResNetBiLSTM, CNN1D_Simple, MLP_Simple




# ---------------------------------------------------------------------------
# Cross-Validation Runner
# ---------------------------------------------------------------------------

def run_cv(model_class, model_name: str, X, y, groups, num_classes: int, num_features: int,
           n_splits: int = 5, epochs: int = 50) -> dict:
    """Run Group K-Fold CV for a given PyTorch model class and return mean/std accuracy."""
    print(f"\n🚀 Benchmarking {model_name} ({n_splits}-Fold CV)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw_full = np.ones(num_classes)
        for i, val in zip(np.unique(y_tr), cw):
            cw_full[i] = val
        w_tensor = torch.tensor(cw_full, dtype=torch.float).to(device)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=128, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=128, shuffle=False,
        )

        model = model_class(num_features, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc = 0.0
        for _ in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    _, pred = model(bx).max(1)
                    total += by.size(0)
                    correct += pred.eq(by).sum().item()
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc = acc

        fold_results.append(best_acc)
        print(f"  Fold {fold+1}: Acc = {best_acc:.2f}%", flush=True)

    return {"Mean Acc": np.mean(fold_results), "Std": np.std(fold_results)}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    print("⏳ Loading all subjects for benchmark...")
    subjects = [f"{i:02d}" for i in range(1, 33)]
    all_X, all_y, all_groups = [], [], []
    class_to_idx: dict = {}

    for subj in subjects:
        subj_dir = os.path.join(PLANTAR_DIR, f"S{subj}")
        if not os.path.isdir(subj_dir):
            continue
        for seq in os.listdir(subj_dir):
            if not seq.startswith("Sequence_"):
                continue
            df = load_and_merge_data(subj, seq)
            if df is None:
                continue
            df_clean = df.dropna(subset=["Class"]).ffill().bfill()
            if not class_to_idx:
                unique_classes = sorted(np.unique(df_clean["Class"].values))
                class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            y_mapped = np.array([class_to_idx[c] for c in df_clean["Class"].values])
            f_cols = [c for c in df_clean.columns if c not in ("Time", "Class", "Action_Name")]
            X_win, y_win, ids_win = create_windows_with_ids(df_clean[f_cols].values, y_mapped, int(subj))
            all_X.append(X_win)
            all_y.append(y_win)
            all_groups.append(ids_win)

    if not all_X:
        print("❌ No data found. Check your .env configuration.")
        return

    X, y, groups = np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_groups)
    N, W, F = X.shape
    X_flat = StandardScaler().fit_transform(X.reshape(-1, F))
    X = X_flat.reshape(N, W, F)
    num_classes = len(class_to_idx)

    final_results: dict = {}

    final_results["MLP_Simple"] = run_cv(
        lambda f, c: MLP_Simple(f, c, W), "MLP_Simple", X, y, groups, num_classes, F
    )
    final_results["CNN1D_Simple"] = run_cv(CNN1D_Simple, "CNN1D_Simple", X, y, groups, num_classes, F)
    final_results["ResNetBiLSTM"] = run_cv(ResNetBiLSTM, "ResNetBiLSTM", X, y, groups, num_classes, F)

    # Random Forest — uses mean-pooled window features as input
    print("\n🌲 Benchmarking Random Forest (5-Fold CV)...")
    gkf = GroupKFold(n_splits=5)
    X_rf = X.mean(axis=1)
    rf_accs = []
    for train_idx, val_idx in gkf.split(X_rf, y, groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
        rf.fit(X_rf[train_idx], y[train_idx])
        rf_accs.append(100.0 * rf.score(X_rf[val_idx], y[val_idx]))
    final_results["RandomForest"] = {"Mean Acc": np.mean(rf_accs), "Std": np.std(rf_accs)}

    out_path = os.path.join(RESULTS_METRICS_DIR, "benchmark_kfold_5fold.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\n✅ Benchmark complete. Results saved → {out_path}")


if __name__ == "__main__":
    main()
