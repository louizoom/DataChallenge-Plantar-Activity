import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

# Titre
nb.cells.append(nbf.v4.new_markdown_cell("""# 📊 Analyse Exploratoire des Données de Capteurs Plantaires (EDA)

Ce notebook (Jupyter Notebook) réalise une analyse exploratoire des signaux de capteurs plantaires et des événements/annotations associés.
Il a été généré via une démarche d'automatisation et suit les étapes suivantes :
1. **Chargement et Intégrité** : Concaténation des signaux et de leurs classes respectives.
2. **Analyse Statistique et Outliers** : Distribution via bar plots et détection des valeurs aberrantes via boxplots.
3. **Visualisation temporelle et spatiale** : Évolution de la pression, analyse inertielle (IMU) et trajectoire 2D du Centre de Pression (CoP).
4. **Comparaison Inter-Classes** : Comparaison de la force totale entre plusieurs actions (ex: Standing vs Walking)."""))

# Importations
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Configuration des graphiques et désactivation de certains warnings mineurs
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Paramètres globaux — chemins relatifs depuis la racine du projet
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), "DataChallenge_donneesGlobales")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")  # ⚠️ nom réel du dossier
EVENTS_DIR = os.path.join(BASE_DIR, "Events")"""))

# 1. Chargement et Intégrité
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Chargement des données et vérification de l'Intégrité

Nous alignons temporellement les classes (annotations) sur les données des signaux à 100 fps en utilisant `Timestamp Start` et `Timestamp End`."""))

nb.cells.append(nbf.v4.new_code_cell("""def load_and_merge_data(subject_id='01', sequence='Sequence_01'):
    \"\"\"
    Charge et fusionne les données de capteurs (insoles.csv) et 
    les annotations (classif.csv) d'un sujet et d'une séquence spécifique.
    Fréquence : 100 fps. Séparateur : ';'
    \"\"\"
    insoles_path = os.path.join(PLANTAR_DIR, f"S{subject_id}", sequence, "insoles.csv")
    classif_path = os.path.join(EVENTS_DIR, f"S{subject_id}", sequence, "classif.csv")
    
    if not os.path.exists(insoles_path):
        print(f"⚠️ Le fichier {insoles_path} est introuvable.")
        return None
        
    if not os.path.exists(classif_path):
        print(f"⚠️ Le fichier {classif_path} est introuvable.")
        return None
        
    print(f"✅ Chargement des données pour le sujet S{subject_id} - {sequence}")
    # Chargement
    df_insoles = pd.read_csv(insoles_path, sep=';')
    df_classif = pd.read_csv(classif_path, sep=';')
    
    # Préparation des colonnes pour les annotations
    df_insoles['Class'] = np.nan
    df_insoles['Action_Name'] = 'Transition/Unknown'
    
    # Alignement temporel des événements
    for _, row in df_classif.iterrows():
        t_start = row['Timestamp Start']
        t_end = row['Timestamp End']
        c_id = row['Class']
        c_name = row['Name']
        
        # Filtre sur la plage temporelle (temps interpolé à 100fps)
        mask = (df_insoles['Time'] >= t_start) & (df_insoles['Time'] <= t_end)
        
        # Affectation
        df_insoles.loc[mask, 'Class'] = c_id
        df_insoles.loc[mask, 'Action_Name'] = c_name
        
    return df_insoles

def display_integrity(df):
    \"\"\"
    Vérifie et affiche les statistiques globales du dataset.
    Affiche le .info(), le .describe() et les valeurs manquantes/infinies.
    \"\"\"
    if df is None: return
    
    print("====== 📋 INFO DU DATASET ======\\n")
    df.info()
    
    print("\\n====== 🧮 STATISTIQUES DESCRIPTIVES (top 15 colonnes) ======\\n")
    display(df.iloc[:, :15].describe()) # Affiche 15 colonnes pour la clarté
    
    print("\\n====== 🛑 VALEURS MANQUANTES ET INFINIES ======\\n")
    # Valeurs manquantes
    missing = df.isna().sum()
    nb_missing = missing[missing > 0]
    print(f"Nombre de colonnes avec des valeurs manquantes : {len(nb_missing)}")
    if len(nb_missing) > 0:
        print(nb_missing.sort_values(ascending=False).head(5))
        
    # Valeurs infinies
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinities = np.isinf(df[numeric_cols]).sum()
    nb_inf = infinities[infinities > 0]
    print(f"Nombre de colonnes avec des valeurs infinies : {len(nb_inf)}")
    if len(nb_inf) > 0:
        print(nb_inf.sort_values(ascending=False).head(5))

# --- DÉMO ---
# À exécuter lorsque les données sont dans l'arborescence
# df_merged = load_and_merge_data('01', 'Seq1')
# display_integrity(df_merged)"""))

# 2. Analyse Statistique
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Analyse Statistique et Outliers

Dans cette partie, on identifie les classes représentées (bar plots) et les potentiels artefacts ou valeurs extrêmes sur les capteurs (box plots)."""))

nb.cells.append(nbf.v4.new_code_cell("""def plot_class_distribution(df):
    \"\"\"Affiche un graphique à barres du nombre d'échantillons par type d'action.\"\"\"
    if df is None: return
    
    plt.figure(figsize=(10, 5))
    counts = df['Action_Name'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    
    plt.title('Distribution des Actions (Nb d\\'échantillons / Frames)', fontweight='bold')
    plt.xlabel('Action/Classe')
    plt.ylabel('Fréquence (Frames)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_outliers(df):
    \"\"\"Utilise des boxplots pour identifier les outliers sur l'IMU et les Pressions.\"\"\"
    if df is None: return
    
    # Identification automatique des colonnes d'accélération (assumant 'acc' et 'x/y/z' ou 'l/r' dans le nom)
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    
    if acc_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[acc_cols], palette='Set2')
        plt.title('Détection des Outliers : Accélérations (IMU)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Accélération')
        plt.tight_layout()
        plt.show()
    
    # Identification automatique de quelques colonnes de pressions afin d'avoir une indication
    press_cols = [c for c in df.columns if 'press' in c.lower() or 'pr_' in c.lower()]
    if press_cols:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df[press_cols[:16]], palette='Set3') # On observe un capteur = 16 colonnes
        plt.title('Détection des Outliers : Capteurs de pression (16 premiers)', fontweight='bold')
        plt.xticks(rotation=90)
        plt.ylabel('Pression brute')
        plt.tight_layout()
        plt.show()

# --- DÉMO ---
# plot_class_distribution(df_merged)
# plot_outliers(df_merged)"""))

# 3. Visualisation temporelle
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Visualisation des Signaux

Nous allons restreindre la visualisation à une seule fenêtre de temps continue (par exemple un bloc ininterrompu d'une classe ciblée comme "Walking") pour éviter la distorsion."""))

nb.cells.append(nbf.v4.new_code_cell("""def visualize_signals_for_action(df, action="Walking"):
    \"\"\"Supervise l'évolution des capteurs et le centre de pression (CoP).\"\"\"
    if df is None: return
    
    # 1. Isoler tous les passages de l'action désirée
    df_action = df[df['Action_Name'] == action].copy()
    if df_action.empty:
        print(f"⚠️ Action '{action}' introuvable.")
        return
        
    # 2. Extraire la PREMIÈRE séquence ininterrompue de cette action (écart temps < 20 ms)
    # df_action['Time'].diff() donnera environ 0.01s (à 100fps). S'il y a un gap de >0.02s, on passe au bloc suivant.
    blocks = (df_action['Time'].diff() > 0.02).cumsum()
    df_window = df_action[blocks == blocks.iloc[0]]
    time_series = df_window['Time']
    
    print(f"🎬 Visualisation de l'action de {time_series.min():.2f}s à {time_series.max():.2f}s (Durée : {time_series.max()-time_series.min():.2f}s)")
    
    # --- 3A. Activation Plantaire (moyenne sur les 16 capteurs G et D) ---
    press_cols_L = [c for c in df_window.columns if ('press' in c.lower() or 'pr_' in c.lower()) and ('_l' in c.lower() or 'left' in c.lower())][:16]
    press_cols_R = [c for c in df_window.columns if ('press' in c.lower() or 'pr_' in c.lower()) and ('_r' in c.lower() or 'right' in c.lower())][:16]
    
    if press_cols_L and press_cols_R:
        mean_L = df_window[press_cols_L].mean(axis=1)
        mean_R = df_window[press_cols_R].mean(axis=1)
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_series, mean_L, label='Moyenne Pressions Gauche', c='forestgreen', lw=2)
        plt.plot(time_series, mean_R, label='Moyenne Pressions Droite', c='crimson', lw=2)
        plt.title(f'Activation Plantaire Moyenne - Action [{action}]', fontweight='bold')
        plt.xlabel('Temps (s)')
        plt.ylabel('Pression Moyenne')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    # --- 3B. IMU (3 axes accélération du pied droit par ex.) ---
    acc_R = [c for c in df_window.columns if 'acc' in c.lower() and ('_r' in c.lower() or 'right' in c.lower())]
    if len(acc_R) >= 3:
        ax, ay, az = acc_R[0], acc_R[1], acc_R[2]
        plt.figure(figsize=(12, 4))
        plt.plot(time_series, df_window[ax], label=f'Axe X ({ax})', alpha=0.9)
        plt.plot(time_series, df_window[ay], label=f'Axe Y ({ay})', alpha=0.9)
        plt.plot(time_series, df_window[az], label=f'Axe Z ({az})', alpha=0.9)
        plt.title(f'Accélération (IMU) Pied Droit - Action [{action}]', fontweight='bold')
        plt.xlabel('Temps (s)')
        plt.ylabel('Accélération')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    # --- 3C. CoP (Trajectoire 2D sur le pied droit) ---
    cop_R = [c for c in df_window.columns if 'cop' in c.lower() and ('_r' in c.lower() or 'right' in c.lower())]
    if len(cop_R) >= 2:
        cx, cy = cop_R[0], cop_R[1]
        plt.figure(figsize=(6, 6))
        # Utilisation de cmp pour identifier le déroulé temporel de la trajectoire
        scatter = plt.scatter(df_window[cx], df_window[cy], c=time_series, cmap='plasma', s=30, label='Mouvements CoP')
        # Ligne de trace sous-jacente 
        plt.plot(df_window[cx], df_window[cy], c='grey', linestyle='-', lw=1, alpha=0.5)
        
        plt.title(f'Trajectoire du Centre de Pression (Pied Droit) - [{action}]', fontweight='bold')
        plt.xlabel(cx)
        plt.ylabel(cy)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Temps (s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- DÉMO ---
# visualize_signals_for_action(df_merged, action="Walking")"""))

# 4. Comparaison des classes
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Comparaison inter-classes

Visualisation des différences de répartition de force globale ("Total Force") entre deux états, par exemple "Standing" et "Walking"."""))

nb.cells.append(nbf.v4.new_code_cell("""def plot_inter_class_comparison(df, class1="Standing", class2="Walking"):
    \"\"\"
    Compare la moyenne de la Force Totale (Total Force) entre 2 actions différentes.
    Si force gauche et droite sont séparables, un diagramme avec Hue est tracé.
    \"\"\"
    if df is None: return
    
    force_cols = [c for c in df.columns if 'total' in c.lower() and 'force' in c.lower()]
    df_compare = df[df['Action_Name'].isin([class1, class2])].copy()
    
    if df_compare.empty or not force_cols:
        print("⚠️ Classes introuvables ou colonnes de 'Total Force' introuvables.")
        return
        
    if len(force_cols) >= 2:
        # Cas où Force Gauche et Force Droite sont distinguées
        fL, fR = force_cols[0], force_cols[1]
        
        # Transformation du format pour Seaborn
        df_melt = pd.melt(df_compare, id_vars=['Action_Name'], value_vars=[fL, fR], 
                          var_name='Côté / Capteur', value_name='Total Force')
        
        plt.figure(figsize=(9, 6))
        sns.barplot(data=df_melt, x='Action_Name', y='Total Force', hue='Côté / Capteur', palette='coolwarm')
        plt.title(f'Comparaison inter-classes : Force Totale ({class1} vs {class2})', fontweight='bold')
        plt.xlabel('Classe d\\'action')
        plt.ylabel('Force Totale Moyenne')
        plt.tight_layout()
        plt.show()
    else:
        # Force Totale Globale (1 colonne)
        fc = force_cols[0]
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_compare, x='Action_Name', y=fc, palette='pastel')
        plt.title(f'Comparaison inter-classes : Force Totale Globale ({class1} vs {class2})', fontweight='bold')
        plt.xlabel('Classe d\\'action')
        plt.ylabel(fc)
        plt.tight_layout()
        plt.show()

# --- DÉMO ---
# plot_inter_class_comparison(df_merged, class1="Standing", class2="Walking")"""))

# Écriture
with open('EDA_Plantar.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook EDA_Plantar.ipynb généré avec succès!")
