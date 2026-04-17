import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Configuration esthétique des graphiques et désactivation des warnings mineurs
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Paramètres globaux correspondant à l'arborescence
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")  # ⚠️ nom réel du dossier
EVENTS_DIR = os.path.join(BASE_DIR, "Events")

# %% [markdown]
# ## 1. Chargement et Intégrité des Données

# %%
def load_and_merge_data(subject_id='01', sequence='Sequence_01'):
    """
    Charge et fusionne les données de capteurs (insoles.csv) et 
    les annotations (classif.csv) d'un sujet et d'une séquence spécifique.
    Fréquence : 100 fps. Séparateur : ';'
    """
    insoles_path = os.path.join(PLANTAR_DIR, f"S{subject_id}", sequence, "insoles.csv")
    classif_path = os.path.join(EVENTS_DIR, f"S{subject_id}", sequence, "classif.csv")
    
    if not os.path.exists(insoles_path):
        print(f"⚠️ Le fichier {insoles_path} est introuvable.")
        return None
        
    if not os.path.exists(classif_path):
        print(f"⚠️ Le fichier {classif_path} est introuvable.")
        return None
        
    print(f"✅ Chargement des données pour le sujet S{subject_id} - {sequence}")
    
    # Chargement (avec séparateur point-virgule comme spécifié)
    df_insoles = pd.read_csv(insoles_path, sep=';')
    df_classif = pd.read_csv(classif_path, sep=';')
    
    # Création des colonnes cibles pour stocker les annotations
    df_insoles['Class'] = np.nan
    df_insoles['Action_Name'] = 'Transition/Unknown'
    
    # Alignement temporel : pour chaque événement, on l'associe à la plage de temps correspondante
    for _, row in df_classif.iterrows():
        t_start = row['Timestamp Start']
        t_end = row['Timestamp End']
        c_id = row['Class']
        c_name = row['Name']
        
        # Création d'un masque pour encadrer chronologiquement l'action
        mask = (df_insoles['Time'] >= t_start) & (df_insoles['Time'] <= t_end)
        
        # Affectation
        df_insoles.loc[mask, 'Class'] = c_id
        df_insoles.loc[mask, 'Action_Name'] = c_name
        
    return df_insoles

def verify_data_integrity(df):
    """
    Affiche les informations globales et détecte les anomalies (valeurs manquantes ou infinies).
    """
    if df is None: return
    
    print("\n====== 📋 INFO DU DATASET ======\n")
    df.info()
    
    print("\n====== 🧮 STATISTIQUES DESCRIPTIVES (Aperçu) ======\n")
    print(df.iloc[:, :15].describe())  # Restriction aux 15 premières colonnes
    
    print("\n====== 🛑 VALEURS MANQUANTES ET INFINIES ======\n")
    # Compte des valeurs manquantes
    missing = df.isna().sum()
    nb_missing = missing[missing > 0]
    print(f"-> Colonnes avec valeurs manquantes : {len(nb_missing)}")
    if len(nb_missing) > 0:
        print(nb_missing.sort_values(ascending=False).head(5))
        
    # Compte des valeurs infinies
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinities = np.isinf(df[numeric_cols]).sum()
    nb_inf = infinities[infinities > 0]
    print(f"-> Colonnes avec valeurs infinies : {len(nb_inf)}")
    if len(nb_inf) > 0:
        print(nb_inf.sort_values(ascending=False).head(5))

# %% [markdown]
# ## 2. Analyse Statistique et Outliers

# %%
def plot_class_distribution(df):
    """Génère un Bar Plot du nombre d'échantillons (frames) pour chaque action."""
    if df is None: return
    
    plt.figure(figsize=(10, 5))
    counts = df['Action_Name'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    
    plt.title('Distribution des Actions (Nb d\'échantillons par action)', fontweight='bold')
    plt.xlabel('Action / Classe')
    plt.ylabel('Fréquence (Frames)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_outliers(df):
    """Utilise des boxplots pour observer la dispersion et déceler les outliers (IMU et Pressions)."""
    if df is None: return
    
    # --- 1. Boxplots: Accélérations (IMU) ---
    # Recherche automatique basée sur des substrings (acc et xyz ou l/r)
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    
    if acc_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[acc_cols], palette='Set2')
        plt.title('Visualisation des Outliers : Accélérations (IMU)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Valeur d\'Accélération')
        plt.tight_layout()
        plt.show()
    
    # --- 2. Boxplots: Pressions (Focus sur le pied gauche 16 valeurs) ---
    press_cols = [c for c in df.columns if ('press' in c.lower() or 'pr_' in c.lower()) and ('_l' in c.lower() or 'left' in c.lower())]
    
    if not press_cols:
        press_cols = [c for c in df.columns if 'press' in c.lower() or 'pr_' in c.lower()]
        
    if press_cols:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df[press_cols[:16]], palette='Set3')
        plt.title('Visualisation des Outliers : Capteurs de pression (Gauche)', fontweight='bold')
        plt.xticks(rotation=90)
        plt.ylabel('Pression (Valeur brute)')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 3. Visualisation Spatio-Temporelle des Signaux

# %%
def visualize_signals_window(df, target_action="Walking"):
    """
    Restreint les données à l'action visée, et trace la moyenne des pressions, les
    trois axes de l'IMU et le chemin du Centre de Pression (CoP) sur un bloc temporel.
    """
    if df is None: return
    
    df_action = df[df['Action_Name'] == target_action].copy()
    if df_action.empty:
        print(f"⚠️ Aucune donnée pour l'action '{target_action}'.")
        return
        
    # Extraction de la PREMIÈRE séquence ininterrompue
    blocks = (df_action['Time'].diff() > 0.02).cumsum()
    df_window = df_action[blocks == blocks.iloc[0]]
    time_arr = df_window['Time']
    
    print(f"\n🎬 Étude temporelle [{target_action}] de {time_arr.min():.2f}s à {time_arr.max():.2f}s")
    
    # --- 3.A. PRESSIONS MOYENNES (G vs D) ---
    press_L = [c for c in df_window.columns if ('press' in c.lower() or 'pr_' in c.lower()) and ('_l' in c.lower() or 'left' in c.lower())][:16]
    press_R = [c for c in df_window.columns if ('press' in c.lower() or 'pr_' in c.lower()) and ('_r' in c.lower() or 'right' in c.lower())][:16]
    
    if press_L and press_R:
        mean_L = df_window[press_L].mean(axis=1)
        mean_R = df_window[press_R].mean(axis=1)
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_arr, mean_L, label='Gauches Moyennes', color='royalblue', lw=2)
        plt.plot(time_arr, mean_R, label='Droites Moyennes', color='firebrick', lw=2)
        plt.title(f'Activation Plantaire Moyenne par pied - {target_action}', fontweight='bold')
        plt.xlabel('Temps chronologique (s)')
        plt.ylabel('Pression moyenne')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    # --- 3.B. IMU (Accélération Pied Droit) ---
    acc_R = [c for c in df_window.columns if 'acc' in c.lower() and ('_r' in c.lower() or 'right' in c.lower())][:3]
    if len(acc_R) == 3:
        ax, ay, az = acc_R
        plt.figure(figsize=(12, 4))
        plt.plot(time_arr, df_window[ax], label=f'Axe X ({ax})', alpha=0.8)
        plt.plot(time_arr, df_window[ay], label=f'Axe Y ({ay})', alpha=0.8)
        plt.plot(time_arr, df_window[az], label=f'Axe Z ({az})', alpha=0.8)
        plt.title(f'Données inertielles (Accélération Pied Droit) - {target_action}', fontweight='bold')
        plt.xlabel('Temps chronologique (s)')
        plt.ylabel('Accélération transmise')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        
    # --- 3.C. CENTRE DE PRESSION (CoP Pied Droit) ---
    cop_R = [c for c in df_window.columns if 'cop' in c.lower() and ('_r' in c.lower() or 'right' in c.lower())][:2]
    if len(cop_R) == 2:
        cx, cy = cop_R
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(df_window[cx], df_window[cy], c=time_arr, cmap='plasma', s=45)
        plt.plot(df_window[cx], df_window[cy], c='grey', linestyle='-', lw=1, alpha=0.4)
        
        plt.title(f'Trajectoire 2D du Centre de Pression (Pied Droit) - {target_action}', fontweight='bold')
        plt.xlabel(cx)
        plt.ylabel(cy)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Temps parcouru (s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 4. Comparaison inter-classes

# %%
def compare_total_force_by_action(df, class1="Standing", class2="Walking"):
    """
    Génère un boxplot et/ou barplot quantitatif de la "Total Force" sur deux ou plusieurs actions comparées.
    """
    if df is None: return
    
    df_comp = df[df['Action_Name'].isin([class1, class2])].copy()
    force_cols = [c for c in df.columns if 'total' in c.lower() and 'force' in c.lower()]
    
    if df_comp.empty or not force_cols:
        print(f"⚠️ Soit les classes n'existent pas, soit la colonne Total Force manque.")
        return
        
    if len(force_cols) >= 2:
        fL, fR = force_cols[0], force_cols[1]
        df_melt = pd.melt(df_comp, id_vars=['Action_Name'], value_vars=[fL, fR], 
                          var_name='Côté / Source', value_name='Valeur Force Totale')
        
        plt.figure(figsize=(9, 6))
        sns.barplot(data=df_melt, x='Action_Name', y='Valeur Force Totale', hue='Côté / Source', palette='Set1', capsize=0.1)
        plt.title(f'Disparité de Force Totale Bilatérale : {class1} vs {class2}', fontweight='bold')
        plt.xlabel('Action ciblée')
        plt.ylabel('Moyenne de la Force Totale')
        plt.tight_layout()
        plt.show()
    else:
        fc = force_cols[0]
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_comp, x='Action_Name', y=fc, palette='pastel')
        plt.title(f'Disparité de Force Totale Globale : {class1} vs {class2}', fontweight='bold')
        plt.xlabel('Action ciblée')
        plt.ylabel(fc)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 5. Execution Principale
# Décommentez et exécutez ces lignes après avoir placé les dossiers de données (S01, etc.) dans l'arborescence ! 

# %%
if __name__ == "__main__":
    # Définissez ici le sujet et la séquence à observer
    TARGET_SUBJECT = '01'
    TARGET_SEQUENCE = 'Sequence_01'
    
    print("\n[====== LANCEMENT DE L'EDA ======]")
    df_merged = load_and_merge_data(TARGET_SUBJECT, TARGET_SEQUENCE)
    
    if df_merged is not None:
        # 1. Intégrité
        verify_data_integrity(df_merged)
        
        # 2. Statistiques et Outliers
        plot_class_distribution(df_merged)
        plot_outliers(df_merged)
        
        # 3. Visualisations sur une action continue
        visualize_signals_window(df_merged, target_action="Walking")
        
        # 4. Comparatif
        compare_total_force_by_action(df_merged, class1="Standing", class2="Walking")
        
    print("\n[====== FIN DE L'EDA ======]")
