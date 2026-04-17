import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# ==== ARCHITETURE MAÎTRE : RESNET 1D + BI-LSTM ==== 

class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        
        # Skip connection path
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
        
        # Point d'entée et augmentation des features
        self.entry_conv = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.entry_bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) # On réduit de moitié le signal 50 -> 25 frames
        
        # Deux blocs résiduels puissants pour isoler la composante dynamique de pression
        self.res_block1 = ResBlock1D(32, 64)
        self.res_block2 = ResBlock1D(64, 128)
        
        # Le chef d'oeuvre Temporel (Bidirectionnel: on lit l'onde à l'endroit ET à l'envers)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
        # Bidirectionnel signifie que l'output fait `hidden_size * 2` (Avant + Retour).
        self.fc1 = nn.Linear(128 * 2, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Permuter pour Conv1D : (Batch, Features, Sequence)
        x = x.permute(0, 2, 1)
        
        x = self.pool(self.relu(self.entry_bn(self.entry_conv(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Repermuter pour LSTM : (Batch, Sequence, Features)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        
        # Global Average Pooling sur le Temps (Plus équilibré que juste prendre la fin)
        # Moyenne sur toute la longueur de la fenêtre (Sequence dim=1)
        avg_pool = torch.mean(lstm_out, dim=1)
        
        out = self.dropout(self.relu(self.fc1(avg_pool)))
        out = self.fc2(out)
        return out

# ==== GESTION DE L'OVERFITTING (EARLY STOPPING) ====
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    print("🚀 Ascension Finale ! Chargement S01 à S08...")
    subjects = [f"{i:02d}" for i in range(1, 33)] # TOUS les 32 sujets !
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
    
    # ⚖️ CALCUL DES POIDS (FIGHT CLASS IMBALANCE)
    # Les classes ultra-rares obtiendront un coefficient gigantesque sur les pertes du réseau!
    class_w = compute_class_weight(class_weight='balanced', classes=np.unique(y_mapped), y=y_mapped)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_weights_tensor = torch.tensor(class_w, dtype=torch.float).to(device)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    X_win, y_win = create_windows(X_scaled, y_mapped, window_size=50, step_size=25)
    print(f"🎯 Fenêtrage Prêt. Matrices: X={X_win.shape} / Num Classes={num_classes}")
    
    X_train, X_val, y_train, y_val = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=128, shuffle=False)
    
    model = ResBiLSTM(num_features=len(feature_cols), num_classes=num_classes).to(device)
    
    print(f"💡 Armure DL Activée : Poids Imbalance & Early Stopping & ResNet-BiLSTM")
    # On ajoute au Cross Entropy la matrice des poids proportionnelle
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopper = EarlyStopping(patience=12, min_delta=0.001)
    
    epochs = 40
    best_acc = 0.0
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
        v_l_avg = val_loss/len(val_loader)
        
        scheduler.step(val_acc)
        
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "master_model_resbilstm.pth"))
            marker = "👑"
            
        print(f"[E{epoch+1:02d}/{epochs}] Loss(W): {train_loss/len(train_loader):.3f} | T-Acc: {train_acc:5.2f}% | Val Loss: {v_l_avg:.3f} | V-Acc: {val_acc:5.2f}% {marker}", flush=True)
        
        early_stopper(v_l_avg)
        if early_stopper.early_stop:
            print("🛑 Early Stopping déclenché. Le modèle stagne, arrêt pour éviter l'Overfitting.")
            break
            
    print(f"\n🔮 EXÉCUTION TERMINÉE ! (T = {time.time() - t0:.1f} s)")
    print(f"📊 Accuracy Absolue : {best_acc:.2f}% (Réseau Imbalance-Proof)")

if __name__ == "__main__":
    main()
