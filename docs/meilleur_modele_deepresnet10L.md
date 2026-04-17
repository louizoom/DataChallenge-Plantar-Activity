# 🏆 Fiche Technique du Meilleur Modèle : DeepResNet-10L

Ce document détaille toutes les caractéristiques du modèle champion de l'étude analytique sur les capteurs plantaires. Il a supplanté l'ensemble des autres architectures testées.

## 1. La Carte d'Identité du Modèle
*   **Nom de l'Architecture** : DeepResNet-10L (Deep Residual Network à 10 Couches)
*   **Tâche** : Classification Multiclasses (31 actions humaines)
*   **Précision Globale (Validation)** : **78.23 %** ⭐
*   **Fichiers de Code Associés** : `train_deep_10L.py` (Script d'entraînement) et `deep_resnet_10L_model.pth` (Poids neuronaux sauvegardés)

---

## 2. L'Architecture Neuronale (Pourquoi "10 Couches" ?)
L'une des plus grandes victoires de la Data Science moderne est le réseau "Résiduel" (ResNet). Généralement, empiler trop de couches fait oublier à la machine les données de départ (Disparition du gradient). Ce modèle utilise des **Skip-Connections** (un pont qui passe par-dessus la couche) pour garder la mémoire du signal originel de la semelle.

Le modèle totalise très exactement **10 couches avec des poids apprenables** (d'où le "10L") :

1.  **Couche Initiale d'Extraction (1 couche)** : 
    *   `Conv1D` (64 filtres, taille de kernel=5). Dégrossit le signal brut.
    *   Suivie d'un `MaxPool` pour réduire l'encombrement temporel de la fenêtre de 50 frames à 25 frames.
2.  **Bloc Résiduel 1 (2 couches)** : Garde la taille du signal à 25 frames, augmente à 128 filtres pour chercher les petits patterns (ex: pose du talon).
3.  **Bloc Résiduel 2 (2 couches)** : Utilise un `Stride=2` (pas de glissement rapide) pour compresser le temps de moitié (25 frames -> 13 frames), et double la réflexion à 256 filtres.
4.  **Bloc Résiduel 3 (2 couches)** : Extraction profonde à 256 filtres sur 13 frames temporelles.
5.  **Bloc Résiduel 4 (2 couches)** : L'extraction finale ultra-abstraite à 512 filtres.
6.  **Extrapolation Finale (1 couche)** : 
    *   On utilise un `Global Average Pooling` pour résumer le sens des 13 "temps" restants en un seul score de signification conceptuelle.
    *   La couche finale : `Linear` (Dense) transforme les 512 filtres d'idées abstraites en **31 probabilités d'actions**. 

---

## 3. Paramètres, Entraînement et Données

L'IA n'est efficace que grâce aux fondations et aux hyperparamètres utilisés pour la contraindre :

*   **Format de la donnée (Sliding Windows)** : Fenêtres temporelles de 50 indices (~0.8 sec d'action par fenêtre), avec un avancement de 25 indices pour assurer la continuité.
*   **Rééquilibrage (Class Imbalance)** : Les classes sur-représentées (comme la Marche) ont été pénalisées mathématiquement par la fonction `compute_class_weight`. Inversement, si le réseau se trompait sur une action rare (ex: Un saut), la perte d'erreur était multipliée par un grand coefficient pour forcer l'apprentissage.
*   **Fonction de Perte (Loss)** : `CrossEntropyLoss` (idéale pour les classifications exclusives de multiples catégories).
*   **Optimiseur (Le Moteur)** : `Adam` combiné à un optimiseur dynamique `ReduceLROnPlateau`. (Si le modèle bloquait depuis 4 époques, l'optimiseur réduisait la brutalité de la correction de 50%).
*   **Époques d'apprentissage** : **50 Époques.**

---

## 4. Pourquoi a-t-il GAGNÉ face aux autres ?

Dans le monde du Machine Learning, il faut trouver la frontière parfaite ("*The Sweet Spot*") entre un modèle stupide et un modèle paranoïaque.

1.  **Face aux modèles basiques (MLP à 42%, CNN1D à 43%)** : Le DeepResNet-10L a triomphé car la donnée biomécanique est trop complexe. Faire un pas ou trébucher demande de l'abstraction que 3 couches ne peuvent physiquement pas calculer. La profondeur (10 couches) était absolument requise.
2.  **Face aux modèles excessifs (Ultimate SE-ResBiLSTM à 77.3%)** : Lorsque nous avons ajouté encore plus de complexité au réseau profond (Attention Squeeze-and-Excitation + Couches temporelles BiLSTM), **le modèle est devenu sur-entraîné**. Il a atteint 95.7% de réussite sur les données qu'il connaissait, mémorisant les tics et le "bruit" des semelles des patients, devenant moins généraliste et moins robuste sur des inconnus. 

**Le DeepResNet-10L est le parfait équilibre entropique de ce projet !**
