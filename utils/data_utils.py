"""
Fonctions utilitaires partagées pour le chargement et le prétraitement des données.
Utilisées par tous les scripts d'entraînement du projet.
"""
import os
import numpy as np
import pandas as pd
from collections import Counter

from .paths import PLANTAR_DIR, EVENTS_DIR


def load_and_merge_data(subject_id, sequence):
    """
    Charge et fusionne les données de capteurs (insoles.csv) avec les
    annotations de classe (classif.csv) pour un sujet et une séquence donnés.

    Args:
        subject_id (str | int): Identifiant du sujet, ex: '01' ou 1.
        sequence (str): Nom de la séquence, ex: 'Sequence_01'.

    Returns:
        pd.DataFrame | None: DataFrame fusionné, ou None si les fichiers sont introuvables.
    """
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
    """
    Crée des fenêtres glissantes sur les données temporelles.
    La classe de chaque fenêtre est déterminée par vote majoritaire.

    Args:
        X (np.ndarray): Données de features, shape (N, num_features).
        y (np.ndarray): Labels correspondants, shape (N,).
        window_size (int): Nombre de frames par fenêtre.
        step_size (int): Pas de glissement entre deux fenêtres.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X_windows, y_windows)
    """
    if step_size is None:
        step_size = window_size // 2

    windows_X = []
    windows_y = []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i: i + window_size]
        win_y = y[i: i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
    return np.array(windows_X), np.array(windows_y)


def create_windows_with_ids(X, y, subject_id, window_size=50, step_size=25):
    """
    Comme create_windows, mais retourne également l'ID du sujet pour chaque fenêtre.
    Utile pour le GroupKFold cross-validation.

    Args:
        X (np.ndarray): Données de features.
        y (np.ndarray): Labels.
        subject_id (int): ID numérique du sujet (pour GroupKFold).
        window_size (int): Taille de la fenêtre.
        step_size (int): Pas de glissement.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (X_windows, y_windows, subject_ids)
    """
    if step_size is None:
        step_size = window_size // 2

    windows_X, windows_y, windows_ids = [], [], []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i: i + window_size]
        win_y = y[i: i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
        windows_ids.append(subject_id)
    return np.array(windows_X), np.array(windows_y), np.array(windows_ids)


def load_all_subjects(n_subjects=32, verbose=True):
    """
    Charge et concatène toutes les données de tous les sujets disponibles.

    Args:
        n_subjects (int): Nombre maximum de sujets à charger (de 01 à n_subjects).
        verbose (bool): Si True, affiche la progression.

    Returns:
        pd.DataFrame | None: DataFrame complet, ou None si aucune donnée.
    """
    subjects = [f"{i:02d}" for i in range(1, n_subjects + 1)]
    all_data = []
    for subj in subjects:
        subj_dir = os.path.join(PLANTAR_DIR, f"S{subj}")
        if not os.path.isdir(subj_dir):
            continue
        for seq in sorted(os.listdir(subj_dir)):
            if seq.startswith("Sequence_"):
                df = load_and_merge_data(subj, seq)
                if df is not None:
                    all_data.append(df)
                    if verbose:
                        print(f"  ✅ S{subj} - {seq} chargé.")
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)


def clean_dataframe(df):
    """
    Nettoie le DataFrame : supprime les lignes sans classe et comble les NaN.

    Args:
        df (pd.DataFrame): DataFrame brut.

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """
    df = df.dropna(subset=['Class'])
    df = df.ffill().bfill()
    return df
