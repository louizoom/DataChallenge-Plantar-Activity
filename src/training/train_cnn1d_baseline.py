"""
Train CNN 1D Baseline — Quick 4-epoch sanity check on S01–S02.

Usage
-----
    python src/training/train_cnn1d_baseline.py
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_here         = os.path.dirname(os.path.abspath(__file__))
_src_root     = os.path.dirname(_here)
_project_root = os.path.dirname(_src_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _src_root)

from utils import load_and_merge_data, create_windows, PLANTAR_DIR
from models import CNN1D_Simple


def main() -> None:
    print("🚀 Loading S01–S02 for CNN 1D Baseline...")
    subjects = ["01", "02"]
    all_data = []
    for subj in subjects:
        subj_dir = os.path.join(PLANTAR_DIR, f"S{subj}")
        if os.path.isdir(subj_dir):
            for seq in os.listdir(subj_dir):
                if seq.startswith("Sequence_"):
                    df = load_and_merge_data(subj, seq)
                    if df is not None:
                        all_data.append(df)

    if not all_data:
        print("❌ No data found.")
        return

    df_clean = pd.concat(all_data, ignore_index=True).dropna(subset=["Class"]).ffill().bfill()
    feature_cols = [c for c in df_clean.columns if c not in ("Time", "Class", "Action_Name")]
    X_raw, y_raw = df_clean[feature_cols].values, df_clean["Class"].values

    unique_classes = np.unique(y_raw)
    class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
    y_mapped       = np.array([class_to_idx[c] for c in y_raw])
    num_classes, num_features = len(unique_classes), len(feature_cols)

    X_scaled     = StandardScaler().fit_transform(X_raw)
    X_win, y_win = create_windows(X_scaled, y_mapped, window_size=50, step_size=25)
    print(f"🪟 Windows: {X_win.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=64, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=64, shuffle=False,
    )

    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model     = CNN1D_Simple(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 4
    print(f"\n🔥 Training CNN1D ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                _, pred = model(bx).max(1)
                total += by.size(0)
                correct += pred.eq(by).sum().item()
        
        print(f"[{epoch+1}/{epochs}] Val Acc: {100.*correct/total:.2f}%")


if __name__ == "__main__":
    main()
