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

# ==== ARCHITECTURE ULTIME : SE-RES-BILSTM ==== 

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block pour signaux temporels (1D).
    Cette couche agit comme une "Attention" qui pondère l'importance
    de chaque capteur ou caractéristique neuronale selon sa pertinence.
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1)
        # Pondération dynamique des features du signal
        return x * y.expand_as(x)

class SEResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(SEResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        
        # Ajout du mécanisme d'attention (SE)
        self.se = SEBlock1D(out_c)
        
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
        # Appliquer l'attention SE juste avant de fusionner
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

class Ultimate_SEResBiLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Ultimate_SEResBiLSTM, self).__init__()
        
        # COUCHE 1 : Entrée (Spatial)
        self.entry_conv = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.entry_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) # 50 frames -> 25 frames
        
        # COUCHES 2 à 9 : Les 4 Blocs Attentionnels (SE-ResNet)
        self.res_block1 = SEResBlock1D(64, 128)
        self.res_block2 = SEResBlock1D(128, 256, stride=2) # 25 -> 13 frames
        self.res_block3 = SEResBlock1D(256, 256)
        self.res_block4 = SEResBlock1D(256, 512)
        
        # LE CERVEAU TEMPOREL (Bi-LSTM) 
        # Lit le signal attentionnel dans les deux sens (Avant/Arrière)
        self.bilstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
        # On ajoute un dropout un poil plus sévère pour éviter le surapprentissage (Max 93% en Train la dernière fois)
        self.dropout = nn.Dropout(0.5)
        
        # COUCHE 10 : L'Unique Couche Dense Finale
        # L'output du BiLSTM fera 128 * 2 = 256
        self.fc_out = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Permuter pour le CNN : (Batch, Features, Sequence)
        x = x.permute(0, 2, 1)
        
        x = self.pool(self.relu(self.entry_bn(self.entry_conv(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Permuter pour le LSTM : (Batch, Sequence, Features)
        # x est de shape (Batch, 512, 13). Re-permute : (Batch, 13, 512)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        
        # Global Average Pooling Temporel sur les 13 frames restantes
        avg_pool = torch.mean(lstm_out, dim=1) # -> (Batch, 256)
        
        x = self.dropout(avg_pool)
        out = self.fc_out(x)
        return out


def main():
    print("🚀 CHARGEMENT DU MODÈLE ULTIME (S01 à S32)...")
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
    
    print("\n🪟 Fenêtrage Spatio Temporel...")
    X_win, y_win = create_windows(X_scaled, y_mapped, window_size=50, step_size=25)
    print(f"Dimensions : X = {X_win.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=256, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"💻 Moteur de calcul activé : {device}")
    
    model = Ultimate_SEResBiLSTM(num_features=num_features, num_classes=num_classes).to(device)
    
    class_w = compute_class_weight(class_weight='balanced', classes=np.unique(y_mapped), y=y_mapped)
    class_weights_tensor = torch.tensor(class_w, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    epochs = 70
    # Le lr est à 0.001 max. Cosine l'amènera doucement vers 0.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    print(f"\n🔥 Démarrage: Entraînement Modèle Ultime ({epochs} Epochs Cosinus)")
    
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
        
        # Le scheduler n'accepte plus d'argument (val_acc) avec CosineAnnealing, il a juste besoin qu'on lui dise qu'une époque est passée
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "ultimate_model_best.pth"))
            marker = "⭐🌟"
        else:
            marker = ""
            
        print(f"[E{epoch+1:02d}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:5.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:5.2f}% {marker}", flush=True)
        
    duration = time.time() - t0
    print(f"\n👑 FINI. Durée Totale : {duration:.1f} secondes.")
    print(f"📊 LE SCORE ABSOLU ULTIME EST : {best_acc:.2f}% (Sauvegardé dans ultimate_model_best.pth)")

if __name__ == "__main__":
    main()
