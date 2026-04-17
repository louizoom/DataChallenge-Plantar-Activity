import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Dossier de sortie : results/ à la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# GRAPHIQUE 1 : COURBE D'APPRENTISSAGE ResBiLSTM
# ============================================
epochs = list(range(1, 41))

train_acc = [
    26.23, 38.11, 41.58, 43.39, 45.06, 46.50, 47.53, 48.55, 49.70, 50.67,
    51.31, 52.57, 53.28, 53.93, 54.75, 55.77, 56.56, 57.37, 57.93, 58.86,
    59.56, 59.84, 61.21, 61.78, 62.56, 63.12, 63.63, 64.76, 65.32, 65.99,
    66.67, 67.54, 67.95, 68.76, 68.87, 69.92, 70.59, 70.94, 71.44, 72.05
]

val_acc = [
    39.07, 42.44, 45.41, 46.56, 47.95, 48.21, 49.54, 50.36, 51.04, 51.31,
    52.43, 52.55, 53.07, 53.53, 54.40, 53.98, 55.00, 55.54, 55.66, 56.31,
    56.59, 57.07, 57.38, 58.33, 58.10, 58.76, 58.99, 59.28, 59.71, 59.70,
    60.29, 60.57, 61.42, 60.75, 61.23, 61.92, 61.86, 62.13, 62.33, 62.79
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
ax.annotate(f'  >> {best_val:.2f}% (E{best_epoch}) BEST',
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 1.5, best_val - 2),
            color='gold', fontsize=11, fontweight='bold')

# Ligne des 60%
ax.axhline(60, color='#ff6b6b', linestyle='--', linewidth=1.2, alpha=0.8, label='Seuil 60%')
ax.text(0.8, 60.5, '60%', color='#ff6b6b', fontsize=9, transform=ax.get_yaxis_transform())

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Courbe d\'Apprentissage — ResBiLSTM (40 Epochs, S01–S08)', fontsize=14, pad=15)
ax.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#444', fontsize=10)
ax.set_xlim(1, 40)
ax.set_ylim(20, 80)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_learning_curve.png'), dpi=150, bbox_inches='tight')
print("✅ results/chart_learning_curve.png sauvegardé")
plt.close()

# ============================================
# GRAPHIQUE 2 : COMPARAISON DES MODÈLES (BARRES)
# ============================================
models = [
    'MLP Dense\n(w=50)',
    'CNN 1D\n(w=20)',
    'CNN 1D\n(w=60)',
    'Conv-LSTM\n(35 epochs)',
    'ResBiLSTM\n(40 epochs)',
    'Random Forest\n(Frame-by-Frame)*',
]
accuracies = [45.83, 47.97, 47.90, 60.98, 62.79, 80.16]
colors = ['#5a5aff', '#5a5aff', '#5a5aff', '#00a884', '#00d4aa', '#888888']

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(models, accuracies, color=colors, width=0.6, edgecolor='none', zorder=3)

# Valeurs au-dessus des barres
for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f'{acc:.1f}%',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='white'
    )

# Ligne 60%
ax.axhline(60, color='#ff6b6b', linestyle='--', linewidth=1.3, alpha=0.9, zorder=4)
ax.text(5.4, 60.8, 'Objectif 60%', color='#ff6b6b', fontsize=9)

# Légende modèles DL vs ML
patch_dl = mpatches.Patch(color='#00a884', label='Deep Learning PyTorch')
patch_ml = mpatches.Patch(color='#888888', label='Machine Learning (baseline*)')
patch_base = mpatches.Patch(color='#5a5aff', label='DL (architecture naïve, 4 epochs)')
ax.legend(handles=[patch_base, patch_dl, patch_ml], loc='upper left',
          facecolor='#1a1a2e', edgecolor='#444', fontsize=9)

ax.set_ylabel('Accuracy de Validation (%)', fontsize=12)
ax.set_title('Comparaison des Architectures — Classification de 31 Actions Plantaires', fontsize=13, pad=15)
ax.set_ylim(0, 95)
ax.grid(True, axis='y', zorder=0)

footnote = '*Le Random Forest bénéficie d\'un data leakage temporel — score non comparable en conditions réelles.'
fig.text(0.01, -0.02, footnote, fontsize=8, color='#aaaaaa', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_model_comparison.png'), dpi=150, bbox_inches='tight')
print("✅ results/chart_model_comparison.png sauvegardé")
plt.close()
