"""
Train SENetBiLSTM — 70 epochs, all 32 subjects, Cosine Annealing.

Usage
-----
    python src/training/train_senet_bilstm.py

Output
------
    models/senet_bilstm_v1.0.pth
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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

_here         = os.path.dirname(os.path.abspath(__file__))
_src_root     = os.path.dirname(_here)
_project_root = os.path.dirname(_src_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _src_root)

from utils import load_and_merge_data, create_windows, PLANTAR_DIR, MODELS_DIR
from models import SENetBiLSTM


def main() -> None:
    print("🚀 Loading all subjects for SENetBiLSTM...")
    subjects = [f"{i:02d}" for i in range(1, 33)]
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
        print("❌ No data found. Check your .env configuration.")
        return

    df_clean = pd.concat(all_data, ignore_index=True).dropna(subset=["Class"]).ffill().bfill()
    feature_cols  = [c for c in df_clean.columns if c not in ("Time", "Class", "Action_Name")]
    X_raw, y_raw  = df_clean[feature_cols].values, df_clean["Class"].values

    unique_classes = np.unique(y_raw)
    class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
    y_mapped       = np.array([class_to_idx[c] for c in y_raw])
    num_classes, num_features = len(unique_classes), len(feature_cols)

    X_scaled      = StandardScaler().fit_transform(X_raw)
    X_win, y_win  = create_windows(X_scaled, y_mapped, window_size=50, step_size=25)
    print(f"🪟 Windows: {X_win.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=256, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=256, shuffle=False,
    )

    device    = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model     = SENetBiLSTM(num_features, num_classes).to(device)
    class_w   = compute_class_weight("balanced", classes=np.unique(y_mapped), y=y_mapped)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float).to(device))
    epochs    = 70
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, checkpoint_path = 0.0, os.path.join(MODELS_DIR, "senet_bilstm_v1.0.pth")
    print(f"\n🔥 Training SENetBiLSTM ({epochs} epochs, Cosine Annealing)...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx); loss = criterion(out, by); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            train_loss += loss.item(); _, pred = out.max(1)
            total += by.size(0); correct += pred.eq(by).sum().item()

        model.eval()
        val_loss, correct_v, total_v = 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx); val_loss += criterion(out, by).item()
                _, pred = out.max(1); total_v += by.size(0); correct_v += pred.eq(by).sum().item()

        val_acc = 100.0 * correct_v / total_v
        scheduler.step()
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc; torch.save(model.state_dict(), checkpoint_path); marker = "⭐🌟"
        print(f"[E{epoch+1:02d}/{epochs}] Train:{100.*correct/total:5.2f}% | Val:{val_acc:5.2f}% {marker}", flush=True)

    print(f"\n👑 Done ({time.time()-t0:.1f}s) | Best Val: {best_acc:.2f}% → {checkpoint_path}")


if __name__ == "__main__":
    main()
