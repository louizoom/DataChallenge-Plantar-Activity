# Script Pecha-Kucha — Challenge Capteurs Plantaires
*~5 minutes | Format : 1 slide par point clé, chaque section = ~1 minute*

---

## 🔷 PARTIE 1 — Transformation des Données (~1 min)
*Slides suggérées : Schéma pipeline (CSV brut → Nettoyage → Fenêtrage 3D), extrait du tableau avant/après fusion.*

### Ce que nous avons fait

Les données brutes comprenaient deux fichiers distincts par séquence :
- **`insoles.csv`** : 51 colonnes de signaux capteurs à 100 FPS (pressions, accélérations, vitesses angulaires, Centre de Pression).
- **`classif.csv`** : Les étiquettes des actions avec leur intervalle temporel précis.

**3 transformations ont été nécessaires :**

1. **Synchronisation Temporelle (Fusion des labels)** : Aucune donnée ne possédait son label directement. Nous avons appliqué un masquage booléen vectorisé sur les `Timestamp Start/End` pour associer à chaque trame de 0.01s son étiquette d'action exacte.

2. **Nettoyage et Standardisation** : Les trames de transition (sans action labelisée) ont été supprimées. Les valeurs manquantes ont été comblées par propagation temporelle (`forward fill`). L'ensemble des 50 features a ensuite été normalisé (`StandardScaler`) pour équilibrer les différentes unités (Newton, g, °/s).

3. **Fenêtrage Spatio-Temporel (Windowing)** : Un geste n'a de sens que dans sa continuité. Nous avons découpé les signaux en **blocs glissants de 50 frames (0.5-0.8 seconde)**, avec un pas de 25 frames. Chaque bloc devient un tenseur 3D `(Batch, Séquence=50, Features=50)`. La classe majoritaire du bloc est retenue comme label cible. Nous avons utilisé un **GroupKFold** sur les 32 sujets pour valider les modèles sans fuite de données d'un individu à l'autre.

---

## 🔷 PARTIE 2 — L'Ascension des Architecture Testées (~1 min)
*Slides suggérées : Pyramide des modèles testés du plus basique au plus complexe, avec barres de progression.*

Nous avons progressivement escaladé la complexité algorithmique pour trouver le meilleur compris sur nos 31 actions humaines complexes :

| # | Modèle | Approche Technologique | Validation Accuracy |
|---|--------|------------------------|---------------------|
| 1 | **MLP Dense** | Réseau simple écrasant le temps | ~42% |
| 2 | **CNN 1D** | Filtrage convolutif temporel | ~43% |
| 3 | **ResBiLSTM** | Couches Résiduelles + LSTM Bidirectionnel | ~46% |
| 4 | **DeepResNet-10L** | Réseau Ultra-Profond (10 Couches) | **78.23%** 🏆 |
| 5 | **SE-ResBiLSTM (Ultime)** | 10 Couches + Mécanisme d'Attention + LSTM | 77.32% (Léger Surapprentissage) |

*Nous avons découvert le parfait équilibre ("Sweet Spot") à l'étape 4 : La donnée demande à être traitée très en profondeur, mais rajouter trop de mécanismes "gadgets" de bout en bout finit par causer de l'overfitting.*

---

## 🔷 PARTIE 3 — L'Architecture Championne : DeepResNet-10L (~1 min 30)
*Slides suggérées : Diagramme de plongée (Couche d'entrée → 4 Blocs → Average Pooling → Dense).*

Notre modèle champion en **PyTorch** a écrasé la barre des 70%. Il est conçu autour d'une architecture résiduelle profonde (10 Couches avec apprentissage de poids) :

### 1. La Vraie Profondeur sans Amnésie ("ResNet")
Nous avions besoin d'empiler des couches car le passage de "la courbure du pied" à l'action mentale "d'un saut" demande plusieurs de niveaux d'abstractions. Le problème ? Trop de couches font "disparaître le gradient". Nous utilisons donc 4 Blocs dotés de **Skip-Connections** : l'information initiale passe *par-dessus* les couches neuronales pour re-fusionner plus loin. Le signal est traité 10 fois sans jamais être déformé ou perdu.

### 2. Du Microscopique au Global (Le Pooling)
Le signal entre sous forme de 50 frames. Convolutions après convolutions (stride=2), l'IA le comprime à 25, puis à 13 frames très denses, avant d'appliquer un **Global Average Pooling**. Ce "résumé temporel" devient le vecteur cognitif sur lequel la dernière couche va parier sa décision.

### 3. La Pondération des Classes ("Weighted Loss")
Certaines actions comme la marche prennent 80% des CSV. L'IA a été empêchée de tricher via la fonction `compute_class_weight` dans sa *CrossEntropyLoss*. Tromper la prédiction d'un "Faux Pas" coûte mathématiquement beaucoup plus cher à l'IA que tromper un "Pas normal".

---

## 🔷 PARTIE 4 — Analyse des Résultats et Impact (~1 min 30)
*Slides suggérées : Courbe d'Accuracy époustouflante grimpant jusqu'à 78%, Matrice ou Points Forts de l'outil.*

### Le Bilan
Grâce à nos modifications architecturales et à l'entraîneur intelligent *ReduceLROnPlateau*, l'IA a convergé brillamment en **50 époques** :

- **Le Score** : **78.23 % d'Accuracy validée scientifiquement** (GroupKFold). 
- **La Prouesse** : Nous ne classifions pas 2 données binaires (Gauche/Droite), mais classifions de manière exclusive **31 classes différentes** de gestes complexes sur des sujets humains que la machine n'a jamais rencontrés !
- **L'Enseignement** : Un algorithme lourd à 12 mécanismes d'attentions (Le modèle Ultime testé à 77.3%) performait mieux sur les données d'entraînements (95%) mais a moins bien "généralisé" dans la vraie vie face à des inconnus. Mieux vaut une IA à la théorie solide (Notre modèle 10L) qu'une IA trop complexe qui apprend un motif par cœur de travers.

### Conclusion

Le défi de transposer des signaux électriques de semelles inertielles en reconnaissance complète de l'activité humaine a été relevé. L'intelligence artificielle a su créer une vraie **carte biomécanique abstraite** là où des techniques classiques échouaient sous les 40%. La fondation pour un déploiement réel médical ou sportif est maintenant solidement en place.

---
*Modèle finalisé et sauvegardé : `deep_resnet_10L_model.pth` (78.23%)*
