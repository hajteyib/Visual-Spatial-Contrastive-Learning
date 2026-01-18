"""
Script simple : Liste TOUTES les relations spatiales avec leurs nombres
Sauvegarde dans complete_relations_analysis/
"""

import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

import config

# Relations spatiales (liste exhaustive)
SPATIAL_KEYWORDS = {
    'on', 'under', 'above', 'below', 'left of', 'right of',
    'near', 'next to', 'inside', 'outside', 'behind', 'in front of',
    'in', 'beside', 'over', 'beneath', 'against', 'at', 'around', 
    'between', 'among', 'across', 'along', 'by'
}

# Charger données
print("Chargement des données...")
with open(config.TRAIN_JSON, 'r') as f:
    data = json.load(f)

# Extraire relations spatiales
relations = []
for img in data:
    for rel in img.get('relationships', []):
        r = rel.get('relationship', '').strip().lower()
        if r in SPATIAL_KEYWORDS:
            relations.append(r)

# Compter
counts = Counter(relations)

# Trier par nombre
sorted_relations = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# Créer dossier
save_dir = os.path.join(config.BASE_DIR, "complete_relations_analysis")

# === FICHIER TEXTE ===
txt_path = os.path.join(save_dir, "spatial_relations_only.txt")
with open(txt_path, 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("RELATIONS SPATIALES - LISTE COMPLÈTE\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total relations spatiales : {len(sorted_relations)}\n")
    f.write(f"Total échantillons : {sum(counts.values())}\n\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Relation':<20} {'Nombre':>10}\n")
    f.write("-" * 50 + "\n")
    for rel, count in sorted_relations:
        f.write(f"{rel:<20} {count:>10,}\n")
    f.write("-" * 50 + "\n")

print(f"✅ Fichier texte : {txt_path}")

# === GRAPHIQUE ===
plt.figure(figsize=(10, 8))
relations_list = [r[0] for r in sorted_relations]
counts_list = [r[1] for r in sorted_relations]

# Colorer selon utilisation actuelle
colors = ['green' if r in ['on', 'under', 'above', 'below', 'left of', 
                            'right of', 'near', 'next to', 'inside', 'outside'] 
          else 'orange' for r in relations_list]

plt.barh(relations_list, counts_list, color=colors, alpha=0.7)
plt.xlabel('Nombre d\'échantillons', fontsize=12, fontweight='bold')
plt.title('Relations Spatiales dans VRD Dataset', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Légende
from matplotlib.patches import Patch
legend = [
    Patch(facecolor='green', alpha=0.7, label='Utilisées actuellement (10)'),
    Patch(facecolor='orange', alpha=0.7, label='Disponibles (non utilisées)')
]
plt.legend(handles=legend, loc='lower right')

plt.tight_layout()
img_path = os.path.join(save_dir, "spatial_relations_only.png")
plt.savefig(img_path, dpi=150, bbox_inches='tight')
print(f"✅ Graphique : {img_path}")
plt.close()

# Afficher résumé
print(f"\n{'='*50}")
print(f"RÉSUMÉ - {len(sorted_relations)} relations spatiales")
print(f"{'='*50}")
for rel, count in sorted_relations:
    print(f"{rel:<20} {count:>10,}")
print(f"{'='*50}")
print(f"{'TOTAL':<20} {sum(counts.values()):>10,}")
