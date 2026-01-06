"""
Analyse Complète du Dataset VRD - TOUTES les Relations

Ce script analyse le dataset complet SANS filtrage pour:
- Découvrir toutes les relations (prédicats) présentes
- Compter leur distribution
- Identifier les relations spatiales vs non-spatiales
- Générer des visualisations
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_all_relations():
    """
    Charge TOUTES les relations du dataset VRD sans filtrage.
    """
    print("=" * 70)
    print("ANALYSE COMPLÈTE - TOUTES LES RELATIONS VRD")
    print("=" * 70)
    
    # Charger le JSON train (contient le plus de données)
    train_json = config.TRAIN_JSON
    
    print(f"\nChargement du dataset complet depuis : {os.path.basename(train_json)}")
    
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    print(f"✅ {len(data)} images chargées")
    
    # Extraire TOUS les prédicats
    all_relations = []
    
    for img in data:
        for rel in img.get('relationships', []):
            # La structure est : rel['relationship'] contient le prédicat
            predicat = rel.get('relationship', '').strip()
            if predicat:
                all_relations.append(predicat)
    
    print(f"✅ {len(all_relations)} relations trouvées au total")
    
    return all_relations

def analyze_relations(relations):
    """
    Analyse la distribution de toutes les relations.
    """
    print(f"\n{'='*70}")
    print(f"DISTRIBUTION DE TOUTES LES RELATIONS")
    print(f"{'='*70}")
    
    # Compter les occurrences
    relation_counts = Counter(relations)
    
    print(f"\nNombre de relations uniques : {len(relation_counts)}")
    print(f"Total d'occurrences : {sum(relation_counts.values())}")
    
    # Afficher toutes les relations triées par fréquence
    print(f"\n{'Relation':<20} {'Count':>8} {'%':>8} {'Barre'}")
    print("-" * 70)
    
    total = sum(relation_counts.values())
    for relation, count in relation_counts.most_common():
        percentage = (count / total) * 100
        bar = '█' * min(int(percentage / 2), 40)  # Max 40 caractères
        print(f"{relation:<20} {count:>8} {percentage:>7.2f}% {bar}")
    
    return relation_counts

def categorize_relations(relation_counts):
    """
    Catégorise les relations en spatiales et non-spatiales.
    """
    print(f"\n{'='*70}")
    print(f"CATÉGORISATION DES RELATIONS")
    print(f"{'='*70}")
    
    # Relations spatiales (celles qu'on utilise + autres possibles)
    spatial_relations = {
        'on', 'under', 'above', 'below', 'left of', 'right of',
        'near', 'next to', 'inside', 'outside', 'in', 'behind',
        'in front of', 'against', 'at', 'over', 'across',
        'along', 'around', 'beside', 'between', 'among', 'beneath'
    }
    
    # Relations actuellement utilisées
    current_relations = {
        'on', 'under', 'above', 'below', 'left of', 'right of',
        'near', 'next to', 'inside', 'outside'
    }
    
    # Classifier
    current_used = {}
    spatial_available = {}
    non_spatial = {}
    
    for rel, count in relation_counts.items():
        rel_lower = rel.lower()
        if rel_lower in current_relations:
            current_used[rel] = count
        elif rel_lower in spatial_relations:
            spatial_available[rel] = count
        else:
            non_spatial[rel] = count
    
    # Afficher résultats
    print(f"\n✅ Relations SPATIALES UTILISÉES (10) :")
    print(f"   Total échantillons : {sum(current_used.values())}")
    for rel, count in sorted(current_used.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {rel:<15} : {count:>6}")
    
    print(f"\n🔍 Relations SPATIALES DISPONIBLES (non utilisées) :")
    print(f"   Total échantillons : {sum(spatial_available.values())}")
    if spatial_available:
        for rel, count in sorted(spatial_available.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   - {rel:<15} : {count:>6}")
    else:
        print(f"   (Aucune autre relation spatiale trouvée)")
    
    print(f"\n❌ Relations NON-SPATIALES ({len(non_spatial)}) :")
    print(f"   Total échantillons : {sum(non_spatial.values())}")
    print(f"   Top 10 :")
    for rel, count in sorted(non_spatial.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {rel:<15} : {count:>6}")
    
    return current_used, spatial_available, non_spatial

def plot_all_relations(relation_counts, save_dir):
    """
    Visualise toutes les relations.
    """
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION DES VISUALISATIONS")
    print(f"{'='*70}")
    
    # Top 20 relations
    top_20 = dict(relation_counts.most_common(20))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    relations = list(top_20.keys())
    counts = list(top_20.values())
    
    colors = ['green' if r.lower() in ['on', 'under', 'above', 'below', 'left of', 
                                        'right of', 'near', 'next to', 'inside', 'outside'] 
              else 'gray' for r in relations]
    
    bars = ax.barh(relations, counts, color=colors, alpha=0.7)
    
    ax.set_xlabel('Nombre d\'occurrences', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Relations dans VRD Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Relations spatiales (utilisées)'),
        Patch(facecolor='gray', alpha=0.7, label='Autres relations')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_relations_top20.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé : {save_path}")
    plt.close()

def plot_categories(current_used, spatial_available, non_spatial, save_dir):
    """
    Visualise la répartition par catégorie.
    """
    categories = ['Spatiales\nUtilisées', 'Spatiales\nDisponibles', 'Non-Spatiales']
    counts = [sum(current_used.values()), 
              sum(spatial_available.values()), 
              sum(non_spatial.values())]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1 : Camembert
    colors_pie = ['#2ecc71', '#3498db', '#95a5a6']
    explode = (0.1, 0, 0)
    
    ax1.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
            colors=colors_pie, explode=explode, textprops={'fontsize': 12})
    ax1.set_title('Répartition par Catégorie', fontsize=14, fontweight='bold')
    
    # Graphique 2 : Barres
    ax2.bar(categories, counts, color=colors_pie, alpha=0.7)
    ax2.set_ylabel('Nombre d\'échantillons', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution par Catégorie', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs sur les barres
    for i, count in enumerate(counts):
        ax2.text(i, count, f'{count:,}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'relations_by_category.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé : {save_path}")
    plt.close()

def save_complete_report(relation_counts, current_used, spatial_available, non_spatial, save_dir):
    """
    Sauvegarde un rapport texte complet.
    """
    report_path = os.path.join(save_dir, 'complete_relations_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT COMPLET - TOUTES LES RELATIONS VRD\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Nombre total de relations uniques : {len(relation_counts)}\n")
        f.write(f"Nombre total d'occurrences : {sum(relation_counts.values())}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("DISTRIBUTION COMPLÈTE (toutes relations)\n")
        f.write("=" * 70 + "\n\n")
        
        for rel, count in relation_counts.most_common():
            f.write(f"{rel:<25} : {count:>6}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CATÉGORISATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Relations SPATIALES UTILISÉES : {len(current_used)}\n")
        f.write(f"  Total échantillons : {sum(current_used.values())}\n\n")
        
        f.write(f"Relations SPATIALES DISPONIBLES : {len(spatial_available)}\n")
        f.write(f"  Total échantillons : {sum(spatial_available.values())}\n\n")
        
        f.write(f"Relations NON-SPATIALES : {len(non_spatial)}\n")
        f.write(f"  Total échantillons : {sum(non_spatial.values())}\n")
    
    print(f"📝 Rapport complet sauvegardé : {report_path}")

def main():
    # Créer dossier de résultats
    analysis_dir = os.path.join(config.BASE_DIR, "complete_relations_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Charger toutes les relations
    all_relations = load_all_relations()
    
    # Analyser
    relation_counts = analyze_relations(all_relations)
    
    # Catégoriser
    current_used, spatial_available, non_spatial = categorize_relations(relation_counts)
    
    # Visualiser
    plot_all_relations(relation_counts, analysis_dir)
    plot_categories(current_used, spatial_available, non_spatial, analysis_dir)
    
    # Rapport
    save_complete_report(relation_counts, current_used, spatial_available, 
                        non_spatial, analysis_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ ANALYSE COMPLÈTE TERMINÉE")
    print(f"{'='*70}")
    print(f"\n📁 Tous les fichiers dans : {analysis_dir}\n")

if __name__ == "__main__":
    main()
