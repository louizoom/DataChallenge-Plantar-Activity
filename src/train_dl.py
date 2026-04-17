import os
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import Counter

warnings.filterwarnings('ignore')

# Arborescence
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")  # ⚠️ nom réel du dossier
EVENTS_DIR = os.path.join(BASE_DIR, "Events")

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
        
        # S'il y a des classes différentes dans la fenêtre de 0.5s, on prend la majoritaire
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
    return np.array(windows_X), np.array(windows_y)

class PlantarCNN1D(nn.Module):
    def __init__(self, num_features, num_classes):
        super(PlantarCNN1D, self).__init__()
        # PyTorch attend pour Conv1d : (Batch, in_channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        
        # Le Flatten transforme (Batch, Channels, Length) -> (Batch, Channels * Length)
        self.flatten = nn.Flatten()
        
        # Taille calculée manuellement: window 50 -> pool -> 25 -> pool -> 12
        self.fc1 = nn.Linear(128 * 12, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x est dimensionné : (Batch, Sequence_Size, Num_Features)
        # on doit le permuter pour PyTorch
        x = x.permute(0, 2, 1) 
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

def main():
    print("🚀 Chargement des Données pour Deep Learning (S01 - S02)...")
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
                        print(f"✅ S{subj} - {seq} chargé.")
    
    if not all_data:
        print("❌ Aucune donnée...")
        return
        
    df_clean = pd.concat(all_data, ignore_index=True).dropna(subset=['Class'])
    df_clean = df_clean.ffill().bfill()
    
    feature_cols = [c for c in df_clean.columns if c not in ['Time', 'Class', 'Action_Name']]
    X_raw = df_clean[feature_cols].values
    y_raw = df_clean['Class'].values
    
    # Redéfinition des classes vers [0, 1 ... N] pour CrossEntropy loss Pytorch
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[c] for c in y_raw])
    num_classes = len(unique_classes)
    print(f"🧩 Nombre de classes uniques détectées : {num_classes}")
    
    # Standardisation sur l'aplat
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Fenêtrage Glissant Temporel (Windowing)
    print("\n🪟 Création des blocs de mémoires spatiaux-temporels (Window = 50 frames)...")
    X_win, y_win = create_windows(X_scaled, y_mapped, window_size=50, step_size=25)
    print(f"Dimensions du Tenseur Final : X = {X_win.shape}, y = {y_win.shape}")
    
    # Train / Val Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
    
    # DataLoader PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Utilisation du GPU Apple Silicon si disponible (MPS), sinon CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Moteur de calcul activé : {device}")
    
    model = PlantarCNN1D(num_features=len(feature_cols), num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 4
    print(f"\n🔥 Entraînement CNN 1D en cours ({epochs} Epochs)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        train_acc = 100. * correct / total
        
        # Phase Validation
        model.eval()
        val_loss = 0
        correct_v = 0
        total_v = 0
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
        print(f"  * Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} | VAL ACCURACY: {val_acc:.2f}%")
        
    print("\n🎯 Entraînement Terminé ! Succès critique CNN 1D PyTorch.")

if __name__ == "__main__":
    main()
