import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Arborescence
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")  # ⚠️ nom réel du dossier
EVENTS_DIR = os.path.join(BASE_DIR, "Events")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

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

def create_windows(X, y, window_size=50, step_size=25):
    windows_X = []
    windows_y = []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i : i + window_size]
        win_y = y[i : i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
    return np.array(windows_X), np.array(windows_y)

# ==== ARCHITECTURE OPTIMISÉE (CONV-LSTM) ====
class ConvLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConvLSTM, self).__init__()
        
        # BLOC CNN 1
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Length / 2
        
        # BLOC CNN 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Length / 4
        
        # BLOC LSTM (Séquences temporelles réduites)
        # Sequence input size: 64 features per timestep
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Partie Dense (Classification)
        self.fc1 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x : (Batch, Sequence, Features) -> (Batch, Features, Sequence) pour Conv1D
        x = x.permute(0, 2, 1)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        # Retour en (Batch, Sequence, Features) pour LSTM
        x = x.permute(0, 2, 1)
        
        # out: Séquences entières, (h_n, c_n): États finaux
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # On ne prend que le 'dernier' hidden_state de l'output : lstm_out[:, -1, :] 
        # (L'état du système à la fin du demi-pas/0.5 seconde)
        final_state = lstm_out[:, -1, :]
        
        out = self.fc1(final_state)
        out = self.relu3(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def main():
    print("🚀 Chargement massif des données (S01 à S05)...")
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
        print("❌ Aucune donnée...")
        return
        
    df_clean = pd.concat(all_data, ignore_index=True).dropna(subset=['Class'])
    df_clean = df_clean.ffill().bfill()
    
    feature_cols = [c for c in df_clean.columns if c not in ['Time', 'Class', 'Action_Name']]
    X_raw = df_clean[feature_cols].values
    y_raw = df_clean['Class'].values
    
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[c] for c in y_raw])
    num_classes = len(unique_classes)
    num_features = len(feature_cols)
    print(f"🧩 Classes uniques: {num_classes}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print("\n🪟 Fenêtrage Spatio Temporel (0.5s par bloc)...")
    X_win, y_win = create_windows(X_scaled, y_mapped, window_size=50, step_size=25)
    print(f"Dimensions : X = {X_win.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=128, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"💻 Moteur de calcul activé : {device}")
    
    model = ConvLSTM(num_features=num_features, num_classes=num_classes).to(device)
    
    # Nous ajoutons des poids de classe Optionnels si besoin, mais utilisons d'abord Adam Standard
    criterion = nn.CrossEntropyLoss()
    # Optimizer performant (Adam avec L2 regularization / weight decay)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    # Scheduler pour affiner la perte sur la fin
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    epochs = 35
    best_acc = 0.0
    print(f"\n🔥 Démarrage: Entraînement Haute Performance Conv-LSTM ({epochs} Epochs)")
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping très utile avec le LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss, correct_v, total_v = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_v += batch_y.size(0)
                correct_v += predicted.eq(batch_y).sum().item()
        
        val_acc = 100. * correct_v / total_v
        
        # Scheduler update using validation accuracy
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_convlstm_model.pth"))
            marker = "⭐"
        else:
            marker = ""
            
        print(f"[{epoch+1:02d}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:5.2f}% | Val Acc: {val_acc:5.2f}% {marker}")
        
    duration = time.time() - t0
    print(f"\n👑 FINI. Durée Totale : {duration:.1f} secondes.")
    print(f"📊 La MEILLEURE Exactitude globale obtenue est : {best_acc:.2f}% (Sauvegardée dans best_convlstm_model.pth)")

if __name__ == "__main__":
    main()
