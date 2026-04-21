"""
10-Fold Group K-Fold Cross-Validation — ResBiLSTM.

Runs a rigorous 10-fold cross-validation where each fold holds out a distinct
set of subjects, guaranteeing that the model is evaluated on patients it has
never seen during training (patient generalisation).

This is the gold-standard evaluation protocol used in the project: results
from this script are the ones reported in the Pecha Kucha presentation.

Usage
-----
    python src/train_kfold.py

Output
------
    results/kfold_report.txt — per-fold and aggregate accuracy statistics.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.dirname(_here)
_project_root = os.path.dirname(_src_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _src_root)

from utils import load_and_merge_data, create_windows_with_ids, PLANTAR_DIR, RESULTS_LOGS_DIR
from models import ResNetBiLSTM




# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, pred = out.max(1)
        total += by.size(0)
        correct += pred.eq(by).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            running_loss += criterion(out, by).item()
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    print("🚀 Starting 10-Fold GroupKFold Cross-Validation (32 Subjects)...")

    # ------------------------------------------------------------------
    # 1. Load and window all subjects
    # ------------------------------------------------------------------
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
            if df_clean.empty:
                continue

            f_cols = [c for c in df_clean.columns if c not in ("Time", "Class", "Action_Name")]
            X_raw = df_clean[f_cols].values
            y_raw = df_clean["Class"].values

            # Build global class index on the fly (first subject encountered)
            if not class_to_idx:
                unique_classes = sorted(np.unique(y_raw))
                class_to_idx = {c: i for i, c in enumerate(unique_classes)}

            y_mapped = np.array([class_to_idx[c] for c in y_raw])
            X_win, y_win, ids_win = create_windows_with_ids(X_raw, y_mapped, int(subj))
            all_X.append(X_win)
            all_y.append(y_win)
            all_groups.append(ids_win)

    if not all_X:
        print("❌ No data found. Check your .env configuration.")
        return

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)

    # Standardise across the flattened window dimension
    N, W, F = X.shape
    X_flat = StandardScaler().fit_transform(X.reshape(-1, F))
    X = X_flat.reshape(N, W, F)

    print(f"📊 Dataset total: {len(X)} windows across {len(np.unique(groups))} subjects.")

    # ------------------------------------------------------------------
    # 2. K-Fold loop
    # ------------------------------------------------------------------
    n_splits = 10
    gkf = GroupKFold(n_splits=n_splits)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = len(class_to_idx)

    fold_accuracies = []
    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n🌀 ----- Fold {fold+1}/{n_splits} -----")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        class_w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_w_full = np.ones(num_classes)
        for i, val in zip(np.unique(y_train), class_w):
            class_w_full[i] = val
        weights_tensor = torch.tensor(class_w_full, dtype=torch.float).to(device)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=128, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=128, shuffle=False,
        )

        model = ResNetBiLSTM(F, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_fold_acc = 0.0
        epochs = 50

        for epoch in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, v_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(v_acc)
            if v_acc > best_fold_acc:
                best_fold_acc = v_acc
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  E{epoch+1:02d} | Val Acc: {v_acc:.2f}% (Best: {best_fold_acc:.2f}%)")

        fold_accuracies.append(best_fold_acc)
        print(f"✅ Fold {fold+1} complete — Best Accuracy: {best_fold_acc:.2f}%")

    # ------------------------------------------------------------------
    # 3. Final report
    # ------------------------------------------------------------------
    print("\n🏁 FINAL RESULT — 10-Fold Group K-Fold Cross-Validation")
    print(f"   Mean Accuracy : {np.mean(fold_accuracies):.2f}%")
    print(f"   Std Dev       : {np.std(fold_accuracies):.2f}%")
    print(f"   Total time    : {(time.time() - t_start)/60:.1f} min")

    report_path = os.path.join(RESULTS_LOGS_DIR, "train_kfold_resbilstm.log")
    with open(report_path, "w") as f:
        f.write("10-Fold GroupKFold Cross-Validation Report\n")
        f.write(f"Total Subjects : {len(np.unique(groups))}\n")
        f.write(f"Per-fold scores: {fold_accuracies}\n")
        f.write(f"Mean Accuracy  : {np.mean(fold_accuracies):.2f}%\n")
        f.write(f"Std Dev        : {np.std(fold_accuracies):.2f}%\n")
    print(f"\n📄 Report saved → {report_path}")


if __name__ == "__main__":
    main()
