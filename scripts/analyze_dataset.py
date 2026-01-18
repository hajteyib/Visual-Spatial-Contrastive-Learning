"""
Script d'analyse complète du dataset de relations spatiales.

Génère des statistiques détaillées et visualisations pour comprendre :
- Distribution des relations spatiales
- Déséquilibre des classes
- Caractéristiques géométriques par relation
- Répartition train/val/test
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import os
from tqdm import tqdm

import config
from dataset import VRDDataset
from utils_geometry import get_spatial_vector

# Configuration pour de belles visualisations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_relation_distribution(dataset, split_name):
    """Analyse la distribution des relations spatiales."""
    print(f"\n{'='*70}")
    print(f"ANALYSE DE LA DISTRIBUTION - {split_name.upper()}")
    print(f"{'='*70}")
    
    # Compter les relations
    relations_count = Counter()
    for sample in dataset.samples:
        relations_count[sample['predicat']] += 1
    
    total = sum(relations_count.values())
    
    # Afficher statistiques
    print(f"\nTotal d'échantillons : {total}")
    print(f"\nDistribution par relation :\n")
    print(f"{'Relation':<15} {'Count':>8} {'Pourcentage':>12} {'Barre'}")
    print("-" * 70)
    
    for relation, count in relations_count.most_common():
        percentage = (count / total) * 100
        bar = '█' * int(percentage / 2)  # Barre proportionnelle
        print(f"{relation:<15} {count:>8} {percentage:>11.2f}% {bar}")
    
    return relations_count, total

def compute_geometric_stats(dataset, split_name):
    """Analyse les caractéristiques géométriques par relation."""
    print(f"\n{'='*70}")
    print(f"ANALYSE GÉOMÉTRIQUE - {split_name.upper()}")
    print(f"{'='*70}")
    
    # Stocker les vecteurs spatiaux par relation
    spatial_features = defaultdict(list)
    
    print(f"\nCalcul des statistiques géométriques...")
    for sample in tqdm(dataset.samples[:5000]):  # Limite à 5000 pour la vitesse
        try:
            # Charger l'image pour obtenir dimensions
            img_path = os.path.join(dataset.img_dir, sample['filename'])
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            img_w, img_h = image.size
            
            # Extraire bboxes
            b_s = sample['sujet_info']['bbox']
            b_o = sample['objet_info']['bbox']
            bbox_s = [b_s['x'], b_s['y'], b_s['x'] + b_s['w'], b_s['y'] + b_s['h']]
            bbox_o = [b_o['x'], b_o['y'], b_o['x'] + b_o['w'], b_o['y'] + b_o['h']]
            
            # Calculer vecteur spatial
            spatial_vec = get_spatial_vector(bbox_s, bbox_o, img_w, img_h)
            
            # Stocker par relation
            relation = sample['predicat']
            spatial_features[relation].append(spatial_vec.numpy())
            
        except Exception as e:
            continue
    
    # Calculer moyennes et std par relation
    print(f"\nCaractéristiques géométriques moyennes par relation :")
    print(f"\n{'Relation':<12} {'dx':>8} {'dy':>8} {'dist':>8} {'sin_a':>8} {'cos_a':>8}")
    print("-" * 70)
    
    stats = {}
    for relation in sorted(spatial_features.keys()):
        features = np.array(spatial_features[relation])
        mean_features = features.mean(axis=0)
        
        stats[relation] = {
            'dx': mean_features[0],
            'dy': mean_features[1],
            'dist_sq': mean_features[2],
            'sin_a': mean_features[3],
            'cos_a': mean_features[4],
            'samples': len(features)
        }
        
        print(f"{relation:<12} {mean_features[0]:>8.3f} {mean_features[1]:>8.3f} "
              f"{mean_features[2]:>8.3f} {mean_features[3]:>8.3f} {mean_features[4]:>8.3f}")
    
    return stats

def plot_distribution(train_count, val_count, test_count, save_dir):
    """Crée un graphique de distribution des relations."""
    relations = list(train_count.keys())
    
    # Données pour le graphique
    train_vals = [train_count[r] for r in relations]
    val_vals = [val_count.get(r, 0) for r in relations]
    test_vals = [test_count.get(r, 0) for r in relations]
    
    # Graphique à barres groupées
    x = np.arange(len(relations))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, train_vals, width, label='Train (70%)', alpha=0.8)
    bars2 = ax.bar(x, val_vals, width, label='Val (20%)', alpha=0.8)
    bars3 = ax.bar(x + width, test_vals, width, label='Test (10%)', alpha=0.8)
    
    ax.set_xlabel('Relation Spatiale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre d\'échantillons', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des Relations Spatiales par Split', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(relations, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'distribution_par_split.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé : {save_path}")
    plt.close()

def plot_imbalance_analysis(relations_count, total, save_dir):
    """Analyse du déséquilibre avec visualisation."""
    relations = list(relations_count.keys())
    counts = list(relations_count.values())
    percentages = [(c / total) * 100 for c in counts]
    
    # Créer figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1 : Camembert
    colors = sns.color_palette("husl", len(relations))
    explode = [0.1 if p < 5 else 0 for p in percentages]  # Détacher les petites parts
    
    ax1.pie(counts, labels=relations, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=explode)
    ax1.set_title('Distribution des Relations (Camembert)', fontsize=14, fontweight='bold')
    
    # Graphique 2 : Barres avec ligne de balance
    colors_bars = ['red' if p < 5 else 'orange' if p < 10 else 'green' for p in percentages]
    bars = ax2.bar(range(len(relations)), counts, color=colors_bars, alpha=0.7)
    ax2.axhline(y=total/len(relations), color='blue', linestyle='--', 
                label=f'Équilibre parfait ({total/len(relations):.0f})', linewidth=2)
    
    ax2.set_xlabel('Relation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nombre d\'échantillons', fontsize=12, fontweight='bold')
    ax2.set_title('Déséquilibre des Classes', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(relations)))
    ax2.set_xticklabels(relations, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter annotations pour classes critiques
    for i, (rel, count, pct) in enumerate(zip(relations, counts, percentages)):
        if pct < 5:
            ax2.text(i, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                    fontsize=8, color='red', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'imbalance_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé : {save_path}")
    plt.close()

def calculate_imbalance_ratio(relations_count):
    """Calcule le ratio de déséquilibre."""
    counts = list(relations_count.values())
    max_count = max(counts)
    min_count = min(counts)
    
    imbalance_ratio = max_count / min_count
    
    print(f"\n{'='*70}")
    print(f"MÉTRIQUES DE DÉSÉQUILIBRE")
    print(f"{'='*70}")
    print(f"\nClasse majoritaire : {max(relations_count, key=relations_count.get)} ({max_count} échantillons)")
    print(f"Classe minoritaire  : {min(relations_count, key=relations_count.get)} ({min_count} échantillons)")
    print(f"Ratio de déséquilibre : {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 100:
        print(f"⚠️  DÉSÉQUILIBRE EXTRÊME ! (>100:1)")
    elif imbalance_ratio > 20:
        print(f"⚠️  DÉSÉQUILIBRE SÉVÈRE ! (>20:1)")
    elif imbalance_ratio > 5:
        print(f"⚠️  DÉSÉQUILIBRE MODÉRÉ (>5:1)")
    else:
        print(f"✅ DÉSÉQUILIBRE ACCEPTABLE (<5:1)")
    
    return imbalance_ratio

def suggest_solutions(imbalance_ratio, relations_count):
    """Propose des solutions basées sur la littérature scientifique."""
    print(f"\n{'='*70}")
    print(f"SOLUTIONS RECOMMANDÉES (Littérature Scientifique)")
    print(f"{'='*70}")
    
    print(f"\n1️⃣  WEIGHTED LOSS (Cross-Entropy Pondérée)")
    print(f"   - Pénaliser davantage les erreurs sur classes rares")
    print(f"   - Facile à implémenter, gains typiques : +5-10%")
    print(f"   - Implémentation : nn.CrossEntropyLoss(weight=class_weights)")
    
    print(f"\n2️⃣  FOCAL LOSS (Lin et al., 2017)")
    print(f"   - Focalise l'apprentissage sur exemples difficiles")
    print(f"   - Très efficace pour déséquilibre extrême")
    print(f"   - Gains typiques : +8-15% sur classes rares")
    
    print(f"\n3️⃣  RESAMPLING DU DATASET")
    print(f"   a) Oversampling des classes rares (duplication)")
    print(f"      - Simple mais risque de sur-apprentissage")
    print(f"   b) Undersampling des classes majoritaires")
    print(f"      - Perte de données utiles")
    print(f"   c) SMOTE (génération synthétique)")
    print(f"      - Meilleur compromis pour données tabulaires")
    
    print(f"\n4️⃣  DATA AUGMENTATION CIBLÉE")
    print(f"   - Augmenter artificiellement les classes rares")
    print(f"   - Variations : rotation, scaling, crop, etc.")
    print(f"   - Gains : +10-20% sur classes rares")
    
    print(f"\n5️⃣  CONTRASTIVE LEARNING AVEC HARD NEGATIVE MINING")
    print(f"   - Sélectionner les paires difficiles pour la loss")
    print(f"   - Adapté à votre approche contrastive")
    print(f"   - Gains : +5-10%")
    
    print(f"\n6️⃣  ENSEMBLE DE MODÈLES")
    print(f"   - Entraîner plusieurs modèles avec différents samplings")
    print(f"   - Voter pour la prédiction finale")
    print(f"   - Gains : +5-15% mais coûteux")
    
    # Calculer les poids de classe recommandés
    total = sum(relations_count.values())
    class_weights = {rel: total / (len(relations_count) * count) 
                     for rel, count in relations_count.items()}
    
    print(f"\n📊 POIDS DE CLASSE RECOMMANDÉS (pour Weighted Loss) :")
    print(f"\nRelation       | Count  | Poids recommandé")
    print("-" * 50)
    for rel, count in sorted(relations_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{rel:<14} | {count:>5} | {class_weights[rel]:>6.2f}")
    
    return class_weights

def main():
    print("=" * 70)
    print("ANALYSE COMPLÈTE DU DATASET DE RELATIONS SPATIALES")
    print("=" * 70)
    
    # Créer dossier de résultats
    analysis_dir = os.path.join(config.BASE_DIR, "dataset_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Charger les datasets
    print(f"\nChargement des datasets...")
    train_ds = VRDDataset(subset='train')
    val_ds = VRDDataset(subset='val')
    test_ds = VRDDataset(subset='test')
    
    # Analyse distribution
    train_count, train_total = analyze_relation_distribution(train_ds, "train")
    val_count, val_total = analyze_relation_distribution(val_ds, "val")
    test_count, test_total = analyze_relation_distribution(test_ds, "test")
    
    # Visualisations
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION DES VISUALISATIONS")
    print(f"{'='*70}")
    
    plot_distribution(train_count, val_count, test_count, analysis_dir)
    plot_imbalance_analysis(train_count, train_total, analysis_dir)
    
    # Métriques de déséquilibre
    imbalance_ratio = calculate_imbalance_ratio(train_count)
    
    # Analyse géométrique (optionnelle, peut être lente)
    user_input = input("\n❓ Effectuer l'analyse géométrique ? (oui/non) [non] : ").strip().lower()
    if user_input == 'oui':
        geometric_stats = compute_geometric_stats(train_ds, "train")
    
    # Solutions recommandées
    class_weights = suggest_solutions(imbalance_ratio, train_count)
    
    # Sauvegarder rapport
    report_path = os.path.join(analysis_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT D'ANALYSE DU DATASET\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total échantillons :\n")
        f.write(f"  - Train : {train_total} (70%)\n")
        f.write(f"  - Val   : {val_total} (20%)\n")
        f.write(f"  - Test  : {test_total} (10%)\n\n")
        f.write(f"Ratio de déséquilibre : {imbalance_ratio:.1f}:1\n\n")
        f.write(f"Poids de classe recommandés :\n")
        for rel, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {rel:<14} : {weight:.2f}\n")
    
    print(f"\n📝 Rapport sauvegardé : {report_path}")
    print(f"📁 Tous les fichiers dans : {analysis_dir}")
    
    print(f"\n{'='*70}")
    print(f"✅ ANALYSE TERMINÉE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
