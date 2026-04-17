"""
Script de vérification rapide du projet.
Vérifie que les chemins de données sont corrects et que les imports fonctionnent.
Usage: python check_project.py
"""
import os
import sys

print("=" * 60)
print("🔍 VÉRIFICATION DU PROJET — Classification Plantaire")
print("=" * 60)

# 1. Vérification des chemins
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLANTAR_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales", "Plantar_activity_trie")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "DataChallenge_donneesGlobales", "Events")

errors = 0

print("\n📁 Chemins des données :")
if os.path.isdir(PLANTAR_DIR):
    subjects = [d for d in os.listdir(PLANTAR_DIR) if d.startswith("S")]
    print(f"  ✅ Plantar_activity_trie/ → {len(subjects)} sujets trouvés")
else:
    print(f"  ❌ Plantar_activity_trie/ INTROUVABLE → {PLANTAR_DIR}")
    errors += 1

if os.path.isdir(EVENTS_DIR):
    subjects_ev = [d for d in os.listdir(EVENTS_DIR) if d.startswith("S")]
    print(f"  ✅ Events/                → {len(subjects_ev)} sujets trouvés")
else:
    print(f"  ❌ Events/ INTROUVABLE → {EVENTS_DIR}")
    errors += 1

# 2. Vérification d'un sujet exemple
if errors == 0:
    subj_dir = os.path.join(PLANTAR_DIR, "S01")
    seqs = [s for s in os.listdir(subj_dir) if s.startswith("Sequence_")]
    print(f"\n📂 S01 : {len(seqs)} séquences (ex: {seqs[0] if seqs else 'N/A'})")
    
    # Vérification d'un fichier insoles.csv
    sample_insoles = os.path.join(subj_dir, seqs[0], "insoles.csv")
    sample_classif = os.path.join(EVENTS_DIR, "S01", seqs[0], "classif.csv")
    if os.path.exists(sample_insoles):
        print(f"  ✅ insoles.csv trouvé")
    else:
        print(f"  ❌ insoles.csv INTROUVABLE → {sample_insoles}")
        errors += 1
    if os.path.exists(sample_classif):
        print(f"  ✅ classif.csv trouvé")
    else:
        print(f"  ❌ classif.csv INTROUVABLE → {sample_classif}")
        errors += 1

# 3. Vérification des imports Python
print("\n📦 Vérification des imports :")
deps = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "torch": "torch (PyTorch)",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}
for mod, name in deps.items():
    try:
        __import__(mod)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} — à installer: pip install {mod}")
        errors += 1

# 4. Vérification des modèles sauvegardés
print("\n💾 Modèles sauvegardés (models/) :")
model_files = [
    "models/deep_resnet_10L_model.pth",
    "models/master_model_resbilstm.pth",
    "models/ultimate_model_best.pth",
    "models/best_convlstm_model.pth",
]
for mf in model_files:
    path = os.path.join(PROJECT_ROOT, mf)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {mf} ({size_mb:.1f} MB)")
    else:
        print(f"  ℹ️  {mf} absent (pas encore entraîné)")

# 5. Résumé
print("\n" + "=" * 60)
if errors == 0:
    print("✅ TOUT EST OK — Le projet est prêt à fonctionner !")
else:
    print(f"❌ {errors} problème(s) détecté(s) — Merci de corriger les erreurs ci-dessus.")
print("=" * 60)

sys.exit(errors)
