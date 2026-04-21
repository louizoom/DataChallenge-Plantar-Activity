"""
Train Random Forest Baseline — Frame-by-frame approach.

Usage
-----
    python src/training/train_random_forest.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_here         = os.path.dirname(os.path.abspath(__file__))
_src_root     = os.path.dirname(_here)
_project_root = os.path.dirname(_src_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _src_root)

from utils import load_and_merge_data, PLANTAR_DIR, RESULTS_METRICS_DIR


def main() -> None:
    print("🚀 Loading S01–S05 for Random Forest Baseline...")
    subjects = [f"{i:02d}" for i in range(1, 6)]
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

    df_clean = pd.concat(all_data, ignore_index=True)
    df_clean = df_clean.dropna(subset=["Class"]).ffill().bfill()

    feature_cols = [col for col in df_clean.columns if col not in ["Time", "Class", "Action_Name"]]
    X = df_clean[feature_cols]
    y = df_clean["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n🌲 Training Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Overall Accuracy: {acc:.2f}")

    unique_classes = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title("Confusion Matrix - Random Forest Baseline")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_METRICS_DIR, "confusion_matrix_rf.png")
    plt.savefig(out_path, dpi=300)
    print(f"📊 Confusion Matrix saved to {out_path}")


if __name__ == "__main__":
    main()
