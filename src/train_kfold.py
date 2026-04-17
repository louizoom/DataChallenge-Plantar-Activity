import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

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

def create_windows_with_ids(X, y, subject_id, window_size=50, step_size=25):
    windows_X = []
    windows_y = []
    windows_ids = []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i : i + window_size]
        win_y = y[i : i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
        windows_ids.append(subject_id)
    return np.array(windows_X), np.array(windows_y), np.array(windows_ids)

# ==== ARCHITECTURE (Réutilisée du train_master) ====
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
            self.downsample = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_c)
            )
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
        self.bilstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(128 * 2, 64)
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

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    return running_loss/len(loader), 100.*correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    return running_loss/len(loader), 100.*correct/total

def main():
    print("🚀 Démarrage du 10-Fold GroupKFold Cross-Validation (32 Sujets)...")
    subjects = [f"{i:02d}" for i in range(1, 33)]
    
    # 1. Chargement et Fenêtrage (une seule fois pour tout le dataset)
    all_X, all_y, all_groups = [], [], []
    scaler = StandardScaler()
    
    # On charge séquence par séquence pour éviter d'exploser la RAM
    for subj in subjects:
        subj_dir = os.path.join(PLANTAR_DIR, f"S{subj}")
        if not os.path.isdir(subj_dir): continue
        for seq in os.listdir(subj_dir):
            if not seq.startswith("Sequence_"): continue
            df = load_and_merge_data(subj, seq)
            if df is None: continue
            
            df_clean = df.dropna(subset=['Class'])
            if df_clean.empty: continue
            df_clean = df_clean.ffill().bfill()
            
            f_cols = [c for c in df_clean.columns if c not in ['Time', 'Class', 'Action_Name']]
            X_raw = df_clean[f_cols].values
            y_raw = df_clean['Class'].values
            
            # Map classes to [0, N-1]
            if not hasattr(main, 'class_to_idx'):
                unique_classes = sorted(np.unique(y_raw))
                main.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
                main.idx_to_class = {i: c for i, c in enumerate(unique_classes)}
            
            y_mapped = np.array([main.class_to_idx[c] for c in y_raw])
            X_win, y_win, ids_win = create_windows_with_ids(X_raw, y_mapped, int(subj))
            
            all_X.append(X_win)
            all_y.append(y_win)
            all_groups.append(ids_win)
    
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    
    # Standardisation sur les fenêtres aplaties (puis remodelage)
    N, W, F = X.shape
    X_flat = X.reshape(-1, F)
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(N, W, F)
    
    print(f"📊 Dataset total : {len(X)} fenêtres sur {len(np.unique(groups))} sujets.")
    
    # 2. Boucle de K-Fold
    n_splits = 10
    gkf = GroupKFold(n_splits=n_splits)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = len(main.class_to_idx)
    num_features = F
    
    fold_accuracies = []
    t_start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n🌀 ----- PLI {fold+1}/{n_splits} -----")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Calcul des poids de classe pour ce pli spécifique
        class_w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_w_full = np.ones(num_classes)
        for i, val in zip(np.unique(y_train), class_w): class_w_full[i] = val
        weights_tensor = torch.tensor(class_w_full, dtype=torch.float).to(device)
        
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=128, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=128, shuffle=False)
        
        model = ResBiLSTM(num_features, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_fold_acc = 0
        epochs = 50
        
        for epoch in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(v_acc)
            
            if v_acc > best_fold_acc:
                best_fold_acc = v_acc
            
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"  E{epoch+1:02d} | Val Acc: {v_acc:.2f}% (B: {best_fold_acc:.2f}%)")
        
        fold_accuracies.append(best_fold_acc)
        print(f"✅ Fin Pli {fold+1} : Meilleure Accuracy = {best_fold_acc:.2f}%")

    # 3. Rapport Final
    print("\n🏁 FINAL RESULT - 10-FOLD GROUP-K-FOLD")
    print(f"Moyenne Accuracy : {np.mean(fold_accuracies):.2f}%")
    print(f"Écart-type : {np.std(fold_accuracies):.2f}%")
    print(f"Temps total : {(time.time() - t_start)/60:.1f} minutes")
    
    with open(os.path.join(RESULTS_DIR, "kfold_report.txt"), "w") as f:
        f.write(f"10-Fold GroupKFold Cross-Validation Report\n")
        f.write(f"Total Subjects: 32\n")
        f.write(f"Individual Folds: {fold_accuracies}\n")
        f.write(f"Mean Accuracy: {np.mean(fold_accuracies):.2f}%\n")
        f.write(f"Std Dev: {np.std(fold_accuracies):.2f}%\n")

if __name__ == "__main__":
    main()
