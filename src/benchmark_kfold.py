import os
import time
import json
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Arborescence
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")  # ⚠️ nom réel du dossier
EVENTS_DIR = os.path.join(BASE_DIR, "Events")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Fonctions Utilitaires ----
def load_and_merge_data(subject_id, sequence):
    if isinstance(subject_id, int):
        subject_id = f"{subject_id:02d}"
    insoles_path = os.path.join(PLANTAR_DIR, f"S{subject_id}", sequence, "insoles.csv")
    classif_path = os.path.join(EVENTS_DIR, f"S{subject_id}", sequence, "classif.csv")
    if not os.path.exists(insoles_path) or not os.path.exists(classif_path):
        return None
    df_insoles = pd.read_csv(insoles_path, sep=';')
    df_classif = pd.read_csv(classif_path, sep=';')
    df_insoles['Class'] = np.nan
    for _, row in df_classif.iterrows():
        mask = (df_insoles['Time'] >= row['Timestamp Start']) & (df_insoles['Time'] <= row['Timestamp End'])
        df_insoles.loc[mask, 'Class'] = row['Class']
    return df_insoles

def create_windows_with_ids(X, y, subject_id, window_size=50, step_size=25):
    windows_X, windows_y, windows_ids = [], [], []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i : i + window_size]
        win_y = y[i : i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
        windows_ids.append(subject_id)
    return np.array(windows_X), np.array(windows_y), np.array(windows_ids)

# ---- Architectures ----

class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.downsample = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride), nn.BatchNorm1d(out_c))
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ResBiLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ResBiLSTM, self).__init__()
        self.entry_conv = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.entry_bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.res_block1 = ResBlock1D(32, 64)
        self.res_block2 = ResBlock1D(64, 128)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.entry_bn(self.entry_conv(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        out = self.dropout(self.relu(self.fc1(avg_pool)))
        out = self.fc2(out)
        return out

class CNN1D_Simple(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN1D_Simple, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # window 50 -> pool -> 25 -> pool -> 12
        self.fc1 = nn.Linear(128 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        out = self.fc2(self.relu(self.fc1(x)))
        return out

class MLP_Simple(nn.Module):
    def __init__(self, num_features, num_classes, window_size=50):
        super(MLP_Simple, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features * window_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# ---- Entraînement ----
def run_cv(model_class, model_name, X, y, groups, num_classes, num_features, n_splits=5, epochs=50):
    print(f"\n🚀 Benchmarking {model_name} (5-Fold CV)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        cw_full = np.ones(num_classes)
        for i, val in zip(np.unique(y_tr), cw): cw_full[i] = val
        w_tensor = torch.tensor(cw_full, dtype=torch.float).to(device)
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=128, shuffle=False)
        
        model = model_class(num_features, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_acc = 0
        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    outputs = model(bx)
                    _, pred = outputs.max(1)
                    total += by.size(0)
                    correct += pred.eq(by).sum().item()
            acc = 100.*correct/total
            if acc > best_acc: best_acc = acc
        
        fold_results.append(best_acc)
        print(f"  Pli {fold+1}: Acc = {best_acc:.2f}%", flush=True)
        
    return {"Mean Acc": np.mean(fold_results), "Std": np.std(fold_results)}

def main():
    # 1. Chargement (Optimisé)
    print("⏳ Chargement des 32 sujets...")
    subjects = [f"{i:02d}" for i in range(1, 33)]
    all_X, all_y, all_groups = [], [], []
    for subj in subjects:
        subj_dir = os.path.join(PLANTAR_DIR, f"S{subj}")
        if not os.path.isdir(subj_dir): continue
        for seq in os.listdir(subj_dir):
            if not seq.startswith("Sequence_"): continue
            df = load_and_merge_data(subj, seq)
            if df is None: continue
            df_clean = df.dropna(subset=['Class']).ffill().bfill()
            if not hasattr(main, 'class_to_idx'):
                unique_classes = sorted(np.unique(df_clean['Class'].values))
                main.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            y_mapped = np.array([main.class_to_idx[c] for c in df_clean['Class'].values])
            f_cols = [c for c in df_clean.columns if c not in ['Time', 'Class', 'Action_Name']]
            X_win, y_win, ids_win = create_windows_with_ids(df_clean[f_cols].values, y_mapped, int(subj))
            all_X.append(X_win); all_y.append(y_win); all_groups.append(ids_win)
    
    X, y, groups = np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_groups)
    N, W, F = X.shape
    X_flat = StandardScaler().fit_transform(X.reshape(-1, F))
    X = X_flat.reshape(N, W, F)
    num_classes = len(main.class_to_idx)
    
    final_results = {}
    
    # -- MLP --
    final_results["MLP_Simple"] = run_cv(lambda f, c: MLP_Simple(f, c, W), "MLP_Simple", X, y, groups, num_classes, F)
    # -- CNN 1D --
    final_results["CNN1D_Simple"] = run_cv(CNN1D_Simple, "CNN1D_Simple", X, y, groups, num_classes, F)
    # -- ResBiLSTM --
    final_results["ResBiLSTM"] = run_cv(ResBiLSTM, "ResBiLSTM", X, y, groups, num_classes, F)
    
    # -- Random Forest (Fast CV) --
    print("\n🌲 Benchmarking Random Forest (5-Fold CV)...")
    gkf = GroupKFold(n_splits=5)
    X_rf = X.mean(axis=1) # On prend la moyenne des pressions sur la fenêtre pour le RF
    rf_accs = []
    for train_idx, val_idx in gkf.split(X_rf, y, groups):
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
        rf.fit(X_rf[train_idx], y[train_idx])
        rf_accs.append(100. * rf.score(X_rf[val_idx], y[val_idx]))
    final_results["RandomForest"] = {"Mean Acc": np.mean(rf_accs), "Std": np.std(rf_accs)}
    
    with open(os.path.join(RESULTS_DIR, "kfold_comparison_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    print("\n✅ Benchmark terminé ! Résultats sauvegardés.")

if __name__ == "__main__":
    main()
