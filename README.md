# 🦶 Benchmarking Plantar — Classification de 31 Actions Humaines

Projet de Deep Learning pour la classification d'activités humaines à partir de capteurs plantaires (insoles + IMU).
L'objectif est d'atteindre la meilleure précision possible sur 31 classes d'actions à partir des signaux temporels.

---

## 📁 Structure du Projet

```
entrainnementIA/
│
├── DataChallenge_donneesGlobales/     # Données brutes (non modifiées)
│   ├── Plantar_activity_trie/         # ⚠️ NOM RÉEL du dossier insoles par sujet
│   │   ├── S01/
│   │   │   ├── Sequence_01/insoles.csv
│   │   │   └── ...
│   │   └── S32/
│   └── Events/                        # Annotations de classification
│       ├── S01/Sequence_01/classif.csv
│       └── ...
│
├── utils/                             # Module partagé (chemins & prétraitement)
│   ├── __init__.py
│   ├── paths.py                       # Config des chemins centralisée
│   └── data_utils.py                  # Fonctions de chargement et fenêtrage
│
├── outputs/                           # Modèles .pth et graphiques sauvegardés
│
├── EDA_Plantar.py                     # Analyse Exploratoire des Données (EDA)
├── EDA_Plantar.ipynb                  # Version notebook de l'EDA
│
├── train_model.py                     # 🌲 Baseline : Random Forest (frame-by-frame)
├── train_dl.py                        # 🧠 CNN 1D simple (S01-S02, 4 epochs)
├── train_optimised.py                 # ⚡ Conv-LSTM (S01-S05, 35 epochs)
├── train_master.py                    # 🎓 ResBiLSTM (tous sujets, 40 epochs)
├── train_ultimate_model.py            # 🚀 SE-Res-BiLSTM (tous sujets, 70 epochs)
├── train_deep_10L.py                  # 👑 DeepResNet-10L (tous sujets, 50 epochs)
│
├── train_kfold.py                     # 🔁 10-Fold GroupKFold (ResBiLSTM)
├── benchmark_kfold.py                 # 📊 5-Fold CV comparaison multi-modèles
├── experiment_runner.py               # 🧪 Benchmark rapide RF + CNN + MLP (S01-S02)
│
├── generate_charts.py                 # 📈 Génère les courbes d'apprentissage
├── build_notebook.py                  # 📓 Génère EDA_Plantar.ipynb
├── check_project.py                   # ✅ Vérifie l'installation et les chemins
│
├── meilleur_modele_deepresnet10L.md   # Documentation du meilleur modèle
├── pecha_kucha_script.md              # Script de présentation
└── README.md                          # Ce fichier
```

---

## 🚀 Démarrage Rapide

### 1. Vérification de l'installation

```bash
python3 check_project.py
```

### 2. Analyse Exploratoire des Données

```bash
python3 EDA_Plantar.py
```

### 3. Entraînement du meilleur modèle (DeepResNet-10L)

```bash
python3 train_deep_10L.py
```

---

## 📊 Résultats du Benchmark

| Modèle | Accuracy Val | Type |
|--------|-------------|------|
| Random Forest (frame-by-frame*) | 80.16% | ML Baseline |
| CNN 1D (fenêtre 20) | 47.97% | DL naïf |
| CNN 1D (fenêtre 60) | 47.90% | DL naïf |
| MLP Dense (fenêtre 50) | 45.83% | DL naïf |
| Conv-LSTM (35 epochs) | ~61% | DL optimisé |
| **ResBiLSTM (40 epochs)** | **62.79%** | **DL avancé** |
| **DeepResNet-10L (50 epochs)** | **~70%+** | **DL champion** |

> *⚠️ Le Random Forest bénéficie d'un data leakage temporel : il voit des frames proches de celles de validation, ce score n'est pas comparable en conditions réelles.*

---

## 🗺️ Pipeline de données

```
insoles.csv  + classif.csv
      ↓ (alignement temporel)
  DataFrame fusionné
      ↓ (nettoyage + StandardScaler)
  Features normalisées
      ↓ (fenêtrage glissant 50 frames / 0.5s)
  Tenseurs (N, 50, F)
      ↓ (DataLoader PyTorch)
  Entraînement du modèle
      ↓ (Early Stopping / Scheduler)
  Modèle .pth sauvegardé
```

---

## ⚙️ Détail des Modèles

### DeepResNet-10L (`train_deep_10L.py`) — Meilleur modèle
- 4 blocs résiduels (ResBlock1D) + Conv d'entrée + GAP + FC
- Entraîné sur les 32 sujets, 50 epochs
- `ReduceLROnPlateau`, gradient clipping, class weights balanced
- Sauvegarde : `deep_resnet_10L_model.pth`

### SE-ResBiLSTM (`train_ultimate_model.py`) — Architecture Ultime
- Squeeze-and-Excitation + ResNet + BiLSTM
- Entraîné sur 32 sujets, 70 epochs avec Cosine Annealing
- Sauvegarde : `ultimate_model_best.pth`

### ResBiLSTM (`train_master.py`)
- ResNet 1D léger + BiLSTM bidirectionnel
- Sauvegarde : `master_model_resbilstm.pth`

---

## 🛠️ Dépendances

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn nbformat
```
