import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Désactivation des warnings mineurs pour nettoyer la sortie console
warnings.filterwarnings('ignore')

# Arborescence
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")  # ⚠️ nom réel du dossier
EVENTS_DIR = os.path.join(BASE_DIR, "Events")

def load_and_merge_data(subject_id, sequence):
    """Charge et synchronise temporellement sur une séquence précise."""
    # On garantit le format S01, S02...
    if isinstance(subject_id, int):
        subject_id = f"{subject_id:02d}"
        
    insoles_path = os.path.join(PLANTAR_DIR, f"S{subject_id}", sequence, "insoles.csv")
    classif_path = os.path.join(EVENTS_DIR, f"S{subject_id}", sequence, "classif.csv")
    
    if not os.path.exists(insoles_path) or not os.path.exists(classif_path):
        return None
        
    df_insoles = pd.read_csv(insoles_path, sep=';')
    df_classif = pd.read_csv(classif_path, sep=';')
    
    df_insoles['Class'] = np.nan
    df_insoles['Action_Name'] = 'Unknown'
    
    for _, row in df_classif.iterrows():
        t_start = row['Timestamp Start']
        t_end = row['Timestamp End']
        c_id = row['Class']
        c_name = row['Name']
        
        mask = (df_insoles['Time'] >= t_start) & (df_insoles['Time'] <= t_end)
        df_insoles.loc[mask, 'Class'] = c_id
        df_insoles.loc[mask, 'Action_Name'] = c_name
        
    return df_insoles

def main():
    print("🚀 Démarrage du pipeline d'entraînement...")
    
    # 1. Chargement des données ciblées (Sujets S01 à S05) pour un test
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
                        print(f"✅ S{subj} - {seq} chargé.")

    if not all_data:
        print("❌ Aucune donnée trouvée !")
        return
        
    df_full = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total frames chargées : {len(df_full)}")
    
    # 2. Preprocessing
    print("\n🧹 Nettoyage des données...")
    # On supprime les frames sans annotation (transition entre deux mouvements)
    df_clean = df_full.dropna(subset=['Class'])
    df_clean = df_clean.ffill().bfill()
    
    # Séparation Features (X) / Label (y)
    feature_cols = [c for c in df_clean.columns if c not in ['Time', 'Class', 'Action_Name']]
    X = df_clean[feature_cols]
    y = df_clean['Class']
    
    print(f"Données prêtes : X shape {X.shape}, y shape {y.shape}")
    
    # 3. Train Test Split
    print("\n🔀 Split Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardization (Mise à l'échelle pour lisser les différences g/dps/N)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Entraînement
    print("\n🧠 Entraînement du modèle (Random Forest Classifier)...")
    print("   -> Ce calcul utilise tous les coeurs (-1) et prend environ 15 à 30 secondes.")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # 5. Évaluation et métriques
    print("\n📈 Prédiction sur l'ensemble de Validation...")
    y_pred = rf.predict(X_test_scaled)
    
    # Mapping numérique vers le nom d'action pour un rapport lisible
    class_mapping = df_clean[['Class', 'Action_Name']].drop_duplicates().set_index('Class')['Action_Name'].to_dict()
    
    # Génération des Target Names triés
    unique_classes_test = np.unique(y_test)
    target_names = [class_mapping.get(c, str(c)) for c in unique_classes_test]
    
    print("\n====== RAPPORT DE CLASSIFICATION ======\n")
    print(classification_report(y_test, y_pred, labels=unique_classes_test, target_names=target_names))
    
    # Sauvegarde de la Matrice de Confusion
    print("\n🎨 Génération de la matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes_test)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Matrice de Confusion de la Forêt Aléatoire (S01-S05)', fontsize=14)
    plt.ylabel('Vraie Action', fontsize=12)
    plt.xlabel('Action Prédite', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Fichier sauvegardé : 'confusion_matrix.png'")
    
    print("\n🎯 Pipeline ML terminé avec succès !")

if __name__ == "__main__":
    main()
