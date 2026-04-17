"""
Chemins centralisés du projet.
Tous les scripts importent depuis ce fichier pour garantir la cohérence.
"""
import os

# Racine du projet (le dossier contenant utils/, src/, etc.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Répertoire des données principales
BASE_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales")

# ⚠️ Le dossier réel s'appelle "Plantar_activity_trie" (pas "Plantar_activity")
PLANTAR_DIR = os.path.join(BASE_DIR, "Plantar_activity_trie")
EVENTS_DIR = os.path.join(BASE_DIR, "Events")

# Répertoire de sortie pour les modèles et graphiques
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
