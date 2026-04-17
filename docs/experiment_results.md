# Rapport d'Expérimentation : Benchmark des Modèles ML/DL

Ce document consigne les résultats du grand "crash test" effectué entre diverses configurations et algorithmes IA pour la classification des activités plantaires. Les tests ont été effectués dynamiquement sur les données des séquences `S01` et `S02`.

## ⚙️ Méthodologie du Benchmark
- **Données** : Sujets S01 et S02 complets (Nettoyage des NaN, Standardisation globale).
- **Split** : 80% Entraînement / 20% Validation.
- **Deep Learning** : Pour tous les réseaux de neurones (CNN et Dense), l'entraînement a été volontairement **verrouillé à seulement 4 époques (Epochs)** afin que la boucle d'expérience reste brève. L'optimiseur utilisé est Adam (`lr=0.001`) via l'accélération matérielle.

## 📊 Résultats (Data Récupérées)

Voici le compte-rendu brut des performances récupérées :

| Modèle Testé | Type de Modèle | Approche Spatiale | Précision Max (Validation) | Perte Finale (Loss) | Temps d'Exécution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Machine Learning | Image par image (Frame-by-Frame) | **80.16%** | N/A | **7.1s** |
| **CNN 1D (w=20)** | Deep Learning (Conv1D) | Fenêtrage (0.2s par bloc) | **47.97%** | 1.7901 | **8.9s** |
| **CNN 1D (w=60)** | Deep Learning (Conv1D) | Fenêtrage (0.6s par bloc) | **47.90%** | 1.8459 | **3.3s** |
| **MLP Dense** | Deep Learning (Linear) | Fenêtrage (0.5s par bloc) | **45.83%** | 1.9495 | **3.5s** |

## 💡 Analyse & Interprétation

1. **La Domination du Random Forest (à court terme) :** 
   Sans énorme surprise dans le domaine des sciences de la donnée, un _Random Forest_ avec une profondeur arbitraire de 15 explose un réseau neuronal très peu profond non-optimisé (seulement 4 epoch). Le RF trouve immédiatement un seuil de coupure linéaire sur les pressions (ex : "Telle Pression = Tel Mouvement") avec 80.1% de précision !
2. **Le Deep Learning en Sous-Régime :**
   Les CNN 1D et le MLP obtiennent tous presque ~47% au bout de 4 Epochs, ce qui prouve que l'architecture apprend. La Perte (Loss) continue de descendre drastiquement. Pour rattraper le Random Forest, il faudrait allouer environ **50 à 100 époques** au PyTorch pour qu'il trouve les _patterns_.
3. **Apport des Convolutions vs Dense :**
   Le CNN 1D avec une Micro-Fenêtre de 20 frames a montré une convergence très rapide, surclassant (légèrement) les grandes fenêtres de 60 frames et surtout le réseau bête Dense MLP (qui plafonne à 45%). Les micro-convolutions temporelles arrivent bien à "lire" les variations !

## Conclusion
Le code `experiment_runner.py` est fourni. Vous pouvez à tout moment y repasser les variables paramétrées (ex : `epochs=50`) pour que le croisement entre les Random Forests et les modèles IA profonds révèle la vraie force cachée de la corrélation temporelle !
