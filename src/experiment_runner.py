import os
import time
import json
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def create_windows(X, y, window_size=50, step_size=None):
    if step_size is None:
        step_size = window_size // 2
    windows_X = []
    windows_y = []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i : i + window_size]
        win_y = y[i : i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
    return np.array(windows_X), np.array(windows_y)

# ==== MODELS ====

class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes, window_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Calculate flattened dimension dynamically
        dummy = torch.zeros(1, num_features, window_size)
        out = self.pool(self.relu(self.conv1(dummy)))
        out = self.pool(self.relu(self.conv2(out)))
        flat_dim = out.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flat_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        out = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return out

class SimpleMLP(nn.Module):
    def __init__(self, num_features, num_classes, window_size):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features * window_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# ==== TRAIN LOGIC ====
def train_pytorch_model(model, train_loader, val_loader, epochs=4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    final_loss = 0
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct_v, total_v, val_loss = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_v += batch_y.size(0)
                correct_v += predicted.eq(batch_y).sum().item()
        
        acc = 100. * correct_v / total_v
        if acc > best_acc: best_acc = acc
        final_loss = val_loss/len(val_loader)
        
    elapsed = time.time() - t0
    return {"Accuracy (%)": round(best_acc, 2), "Final Val Loss": round(final_loss, 4), "Time (s)": round(elapsed, 1)}

def main():
    print("🚀 Démarrage Benchmark. Chargement S01-S02...")
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
                        
    df_clean = pd.concat(all_data, ignore_index=True).dropna(subset=['Class'])
    df_clean = df_clean.ffill().bfill()
    
    feature_cols = [c for c in df_clean.columns if c not in ['Time', 'Class', 'Action_Name']]
    X_raw = df_clean[feature_cols].values
    y_raw = df_clean['Class'].values
    
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[c] for c in y_raw])
    num_classes = len(unique_classes)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    num_features = len(feature_cols)
    
    results = {}
    
    # === EXP 1: RandomForest (Frame by Frame) ===
    print("🔥 EXP 1/4: RandomForest (Baseline)")
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_mapped, test_size=0.2, random_state=42)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    acc_rf = 100. * rf.score(X_te, y_te)
    tr_time_rf = time.time() - t0
    results["Random Forest (Frame-By-Frame)"] = {"Accuracy (%)": round(acc_rf, 2), "Final Val Loss": "N/A", "Time (s)": round(tr_time_rf, 1), "Type": "Machine Learning"}
    
    # Setup Data Loaders for Torch Windows
    def get_loader(X_s, y_m, window):
        X_w, y_w = create_windows(X_s, y_m, window_size=window, step_size=window//2)
        X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long)), batch_size=64, shuffle=False)
        return train_loader, val_loader

    # === EXP 2: CNN 1D (Window=20 frames) ===
    print("🔥 EXP 2/4: CNN 1D (0.2s / Fenêtre=20)")
    train_dl, val_dl = get_loader(X_scaled, y_mapped, window=20)
    model2 = CNN1D(num_features, num_classes, window_size=20)
    res2 = train_pytorch_model(model2, train_dl, val_dl, epochs=4)
    res2["Type"] = "DL - CNN 1D"
    results["CNN 1D (Fenêtre=20)"] = res2
    
    # === EXP 3: CNN 1D (Window=60 frames) ===
    print("🔥 EXP 3/4: CNN 1D (0.6s / Fenêtre=60)")
    train_dl, val_dl = get_loader(X_scaled, y_mapped, window=60)
    model3 = CNN1D(num_features, num_classes, window_size=60)
    res3 = train_pytorch_model(model3, train_dl, val_dl, epochs=4)
    res3["Type"] = "DL - CNN 1D"
    results["CNN 1D (Fenêtre=60)"] = res3

    # === EXP 4: MLP (Window=50 frames) ===
    print("🔥 EXP 4/4: MLP (0.5s / Fenêtre=50)")
    train_dl, val_dl = get_loader(X_scaled, y_mapped, window=50)
    model4 = SimpleMLP(num_features, num_classes, window_size=50)
    res4 = train_pytorch_model(model4, train_dl, val_dl, epochs=4)
    res4["Type"] = "DL - Réseau Dense"
    results["MLP Dense (Fenêtre=50)"] = res4
    
    print("\n✅ Benchmark terminé !")
    with open(os.path.join(RESULTS_DIR, "experiments_out.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
