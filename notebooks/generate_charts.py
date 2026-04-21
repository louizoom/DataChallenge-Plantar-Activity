import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Project utilities — resolve output path from .env / environment variables
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import RESULTS_CHARTS_DIR

# Style général
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#1a1a2e',
    'figure.facecolor': '#0f0f1a',
    'axes.edgecolor': '#444',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': '#333355',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})

# ============================================
# GRAPHIQUE 1 : COURBE D'APPRENTISSAGE DeepResNet-10L (Champion)
# ============================================
epochs = list(range(1, 51))

train_acc = [
    40.82, 48.47, 51.48, 53.92, 56.17, 58.22, 59.85, 61.78, 63.54, 64.88,
    66.55, 68.01, 69.43, 70.68, 71.83, 72.96, 73.77, 74.87, 75.72, 76.61,
    77.29, 77.98, 78.58, 79.21, 79.68, 80.34, 80.69, 81.11, 81.43, 81.83,
    82.25, 82.63, 82.89, 83.16, 83.50, 83.56, 83.86, 84.10, 84.40, 84.36,
    84.85, 85.02, 85.03, 85.39, 85.34, 85.59, 91.22, 92.62, 92.88, 93.10
]

val_acc = [
    46.61, 49.87, 52.72, 53.98, 56.38, 57.09, 59.18, 60.48, 61.54, 62.59,
    63.97, 64.53, 65.73, 66.17, 66.67, 67.48, 67.54, 68.23, 68.87, 69.66,
    69.52, 70.50, 70.21, 70.70, 71.44, 71.58, 71.82, 71.48, 71.63, 72.03,
    72.35, 72.32, 72.69, 72.28, 72.07, 72.47, 72.87, 73.03, 73.61, 73.27,
    73.88, 73.62, 73.41, 73.26, 73.71, 73.78, 77.81, 78.07, 77.95, 78.23
]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, train_acc, color='#7b68ee', linewidth=2, label='Train Accuracy')
ax.fill_between(epochs, train_acc, alpha=0.1, color='#7b68ee')

ax.plot(epochs, val_acc, color='#00d4aa', linewidth=2.5, label='Val Accuracy', zorder=5)
ax.fill_between(epochs, val_acc, alpha=0.12, color='#00d4aa')

# Annotation du pic
best_epoch = val_acc.index(max(val_acc)) + 1
best_val = max(val_acc)
ax.scatter([best_epoch], [best_val], color='gold', s=100, zorder=10)
ax.annotate(f'  >> {best_val:.2f}% (E{best_epoch}) BEST (Group K-Fold)',
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 1.5, best_val - 2),
            color='gold', fontsize=11, fontweight='bold')

# Ligne des 60%
ax.axhline(60, color='#ff6b6b', linestyle='--', linewidth=1.2, alpha=0.8, label='Seuil 60%')
ax.text(0.8, 60.5, 'Meilleure Baseline (~60%)', color='#ff6b6b', fontsize=9, transform=ax.get_yaxis_transform())

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Courbe d\'Apprentissage — DeepResNet-10L (Champion, S01–S32)', fontsize=14, pad=15)
ax.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#444', fontsize=10)
ax.set_xlim(1, 50)
ax.set_ylim(30, 100)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_CHARTS_DIR, 'learning_curve_deepresnet10l.png'), dpi=150, bbox_inches='tight')
print("✅ results/charts/learning_curve_deepresnet10l.png saved")
plt.close()

# ============================================
# GRAPHIQUE 2 : COMPARAISON DES MODÈLES (BARRES)
# ============================================
models = [
    'Random Forest\n(Group K-Fold)',
    'MLP Dense\n(Group K-Fold)',
    'CNN 1D\n(Group K-Fold)',
    'ResBiLSTM\n(Group K-Fold)',
    'SE-ResBiLSTM\n(Ultime)',
    'DeepResNet-10L\n(Champion)',
]
# Valeurs validées dans le Pecha Kucha et benchmarks K-Fold
accuracies = [42.4, 42.1, 43.3, 46.5, 77.3, 78.2]
colors = ['#5a5aff', '#5a5aff', '#5a5aff', '#00a884', '#ffcc00', '#ff9900']

fig, ax = plt.subplots(figsize=(12, 6))

# Création des barres
bars = ax.bar(models, accuracies, color=colors, width=0.6, edgecolor='none', zorder=3)

# Valeurs au-dessus des barres
for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f'{acc:.1f}%',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='white'
    )

# Ligne d'objectif 60%
ax.axhline(60, color='#ff6b6b', linestyle='--', linewidth=1.3, alpha=0.9, zorder=4)
ax.text(5.4, 60.8, 'Objectif 60%', color='#ff6b6b', fontsize=9)

# Légende
patch_base = mpatches.Patch(color='#5a5aff', label='Baselines (Group K-Fold)')
patch_adv = mpatches.Patch(color='#00a884', label='Deep Learning Avancé')
patch_champion = mpatches.Patch(color='#ff9900', label='Modèles Champions (70%+)')

ax.legend(handles=[patch_base, patch_adv, patch_champion], loc='upper left',
          facecolor='#1a1a2e', edgecolor='#444', fontsize=9)

ax.set_ylabel('Accuracy de Validation Scientifique (%)', fontsize=12)
ax.set_title('Comparaison Finale — Validation Croisée par Sujets (Group K-Fold)', fontsize=13, pad=15)
ax.set_ylim(0, 95)
ax.grid(True, axis='y', zorder=0)

footnote = 'Note : Le score Group K-Fold garantit la capacité de généralisation à de nouveaux patients.'
fig.text(0.01, -0.02, footnote, fontsize=8, color='#aaaaaa', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_CHARTS_DIR, 'model_comparison_kfold.png'), dpi=150, bbox_inches='tight')
print("✅ results/charts/model_comparison_kfold.png saved")
plt.close()
