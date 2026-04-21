"""
Quick Experiment Runner — Baseline Comparison (S01–S02).

Runs four quick experiments on a small subset (S01–S02) to benchmark
architectures with minimal compute time:

    1. Random Forest (frame-by-frame, temporal data leakage ⚠️)
    2. CNN 1D — window = 20 frames (0.2 s)
    3. CNN 1D — window = 60 frames (0.6 s)
    4. MLP Dense — window = 50 frames (0.5 s)

All deep-learning models train for 4 epochs only. Results are saved to
``results/experiments_out.json``.

⚠️  These results use a random train/test split (no GroupKFold). They are
    indicative only and should not be compared directly with the validated
    Group K-Fold scores from ``benchmark_kfold.py``.

Usage
-----
    python src/experiment_runner.py

Output
------
    results/experiments_out.json — accuracy, loss and runtime per experiment.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.dirname(_here)
_project_root = os.path.dirname(_src_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _src_root)

from utils import load_and_merge_data, create_windows, PLANTAR_DIR, RESULTS_METRICS_DIR
from models import CNN1D_Dynamic, MLP_Simple




# ---------------------------------------------------------------------------
# Training Helper
# ---------------------------------------------------------------------------

def train_pytorch_model(model, train_loader, val_loader, epochs: int = 4) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    final_loss = 0.0
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        correct_v, total_v, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item()
                _, pred = out.max(1)
                total_v += by.size(0)
                correct_v += pred.eq(by).sum().item()

        acc = 100.0 * correct_v / total_v
        if acc > best_acc:
            best_acc = acc
        final_loss = val_loss / len(val_loader)

    return {
        "Accuracy (%)": round(best_acc, 2),
        "Final Val Loss": round(final_loss, 4),
        "Time (s)": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    print("🚀 Quick Benchmark — Loading S01–S02...")
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
        print("❌ No data found. Check your .env configuration.")
        return

    df_clean = pd.concat(all_data, ignore_index=True).dropna(subset=["Class"]).ffill().bfill()

    feature_cols = [c for c in df_clean.columns if c not in ("Time", "Class", "Action_Name")]
    X_raw = df_clean[feature_cols].values
    y_raw = df_clean["Class"].values

    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[c] for c in y_raw])
    num_classes = len(unique_classes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    num_features = len(feature_cols)

    results: dict = {}

    # Exp 1: Random Forest (frame-by-frame)
    print("🔥 EXP 1/4: Random Forest (Baseline — frame-by-frame)")
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_mapped, test_size=0.2, random_state=42)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    acc_rf = 100.0 * rf.score(X_te, y_te)
    results["Random Forest (Frame-By-Frame)"] = {
        "Accuracy (%)": round(acc_rf, 2),
        "Final Val Loss": "N/A",
        "Time (s)": round(time.time() - t0, 1),
        "Type": "Machine Learning",
    }

    def get_loader(X_s, y_m, window):
        X_w, y_w = create_windows(X_s, y_m, window_size=window, step_size=window // 2)
        X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
        tl = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=64, shuffle=True,
        )
        vl = DataLoader(
            TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long)),
            batch_size=64, shuffle=False,
        )
        return tl, vl

    # Exp 2: CNN 1D (window=20)
    print("🔥 EXP 2/4: CNN 1D (window=20 frames)")
    tl, vl = get_loader(X_scaled, y_mapped, 20)
    res2 = train_pytorch_model(CNN1D_Dynamic(num_features, num_classes, 20), tl, vl)
    res2["Type"] = "DL - CNN 1D"
    results["CNN 1D (window=20)"] = res2

    # Exp 3: CNN 1D (window=60)
    print("🔥 EXP 3/4: CNN 1D (window=60 frames)")
    tl, vl = get_loader(X_scaled, y_mapped, 60)
    res3 = train_pytorch_model(CNN1D_Dynamic(num_features, num_classes, 60), tl, vl)
    res3["Type"] = "DL - CNN 1D"
    results["CNN 1D (window=60)"] = res3

    # Exp 4: MLP (window=50)
    print("🔥 EXP 4/4: MLP Dense (window=50 frames)")
    tl, vl = get_loader(X_scaled, y_mapped, 50)
    res4 = train_pytorch_model(MLP_Simple(num_features, num_classes, 50), tl, vl)
    res4["Type"] = "DL - Dense Network"
    results["MLP Dense (window=50)"] = res4

    out_path = os.path.join(RESULTS_METRICS_DIR, "baseline_experiments.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Benchmark complete. Results saved → {out_path}")


if __name__ == "__main__":
    main()
