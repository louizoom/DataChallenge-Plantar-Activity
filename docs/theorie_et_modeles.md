# Analyse Théorique : Lien entre les Cours de Data Science et l'Architecture du Projet

Ce document met en lumière les fondamentaux théoriques issus des cours de Data Sciences (enseignés par Benjamin Allaert, José Mennesson et Arthur Louchart) et justifie les choix architecturaux et méthodologiques réalisés au sein de ce projet de classification d'activités plantaires.

---

## 1. La Validation des Modèles : Éviter les Biais 
*(Réf: `Main Document.pdf` - Arthur Louchart)*

Dans son cours, la nécessité de la **Validation Croisée (Cross Validation)** est abordée pour *"réduire le sur-apprentissage (overfitting) et évaluer de manière fiable"*. 

- **Le problème initial :** Pour notre premier modèle (la Baseline Random Forest), nous avions effectué un simple découpage `Train/Test` (Image par image). Le problème observé (score artificiellement élevé à ~80%) est ce que le cours définit comme un biais de découpage aléatoire. Les fenêtres temporelles (frames) de la même seconde se retrouvent réparties à la fois dans l'entraînement et dans le test, ce qui fausse totalement le résultat.
- **La solution structurée (Le Group K-Fold) :** Le cours montre un schéma appliquant la séparation par **Sujets** (*"Sujet 1, Sujet 2, Sujet 3..."*). C'est pour cette raison qu'ont été créés les scripts `train_kfold.py` et `benchmark_kfold.py`. En appliquant la stratégie **Groupe K-Fold**, nous nous sommes assurés que les fenêtres d'un même patient (S01, S02...) ne soient **jamais** fracturées entre l'entraînement et l'évaluation. Notre évaluation garantit que le réseau est capable de généraliser son apprentissage sur un **nouveau patient** inconnu.

---

## 2. Transition vers le Deep Learning
*(Réf: `Introduction Course Presentation` & `Course Part B` - Benjamin Allaert)*

Le cours d'introduction oppose conceptuellement l'apprentissage Machine Learning (ML) au Deep Learning (DL). Le Machine learning "classique" atteint rapidement ses limites sur des séries temporelles pures si on ne l'aide pas avec beaucoup de calcul de features manuelles.

- **Le plafond de verre du ML :** Le réseau dense (MLP) et le Random Forest n'avaient pas de *"conscience spatio-temporelle"*. Ils évaluent les observations indépendamment.
- **L'application du Deep Learning :** Le `Course Part B` indique qu'il faut utiliser des architectures plus complexes et profondes, équipées de fonctions non-linéaires (*"Activation functions like ReLU to make it possible to solve more complex problems"*). C'est pourquoi nous sommes passés aux **Réseaux Convolutifs (CNN 1D)**. Les Convolutional Layers analysent ("filtrent") la donnée multivariée (les 50 capteurs des semelles) à travers la dimension temporelle via une fenêtre glissante (`Window_Size = 50`) pour extraire des signatures dynamiques invisibles en Machine Learning classique.

---

## 3. La Fonction de Perte (Loss Function)
*(Réf: `Course Part B` - Benjamin Allaert)*

Le cours est très clair à ce sujet (Slide 9) : *"We only use the cross-entropy loss function in classification task"*.

- **Choix de l'optimisation :** Notre challenge est une tâche de classification multiclasse de comportements humains. Pour que notre algorithme "apprenne" de ses erreurs (backpropagation), l'optimiseur dans tous les scripts utilise le module canonique recommandé : `nn.CrossEntropyLoss()`.
- **Adaptation au contexte (L'équilibrage) :** Étant donné qu'une action "Transition" ou "Saut" est incroyablement plus rare que l'action "Marche" présente en continu, nous avons complété la théorie en injectant un "poids de classe" (`class_weight='balanced'`) dans cette fonction CrossEntropy. Cela informe le modèle mathématique que l'erreur sur une classe minoritaire doit être pénalisée plus sévèrement.

---

## 4. L'Environnement PyTorch
*(Réf: `Introduction to PyTorch` - José Mennesson)*

L'initiation aux tenseurs (`Tensors`) et à la syntaxe PyTorch est la colonne vertébrale technologique.

- **Développement Natif :** Le projet a complètement substitué les objets DataFrames de Pandas (faits pour le nettoyage) au profit du `TensorDataset` et du `DataLoader` de PyTorch pour l'entraînement. Les briques de code (`nn.Module`, `optim.Adam`, etc.) sont au cœur de l'implémentation.
- **Accélération Matérielle :** Nos travaux gèrent automatiquement le transfert des Tensors vers la mémoire GPU ou la puce Mac (`MPS`), s'affranchissant de la lourdeur des calculs mentionnée en cours.

---

## 🌟 Synthèse : L'Architecture du Modèle Champion (DeepResNet-10L)

Le `Course Part B` prévient de la complexité des Deep Neural Networks : ajouter de nombreuses couches permet une compréhension asymétrique profonde mais s'accompagne d'un grave risque : si le gradient d'erreur s'effondre en cours de rétropropagation, les premières couches n'apprennent plus rien (le *Vanishing Gradient*).

En utilisant un design **Résiduel (ResBlocks)**, nous avons mis en pratique la solution architecturale pour contrer ce phénomène. Le "shortcut" (ou bypass de l'identité) ajouté au sein de nos blocs `ResBlock1D` permet de concevoir une architecture très profonde (**10 Couches !**) sans étouffer le flux du gradient.

**C'est cette architecture (le `DeepResNet-10L`) qui domine le comparatif car elle allie :**
1. **L'Extraction de motifs (CNN 1D)** des appuis plantaires dans le temps.
2. **Une conception anti-obstruction (ResNet)** qui s'affranchit du blocage des CNN trop empilés.
3. **L'Évaluation rigoureuse non-biaisée** du *Group K-Fold* imposé par la structure expérimentale vue en cours.
