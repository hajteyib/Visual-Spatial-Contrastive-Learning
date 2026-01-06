"""
Évaluation Baseline - ResNet-50 ImageNet sans Entraînement Contrastif

Ce script évalue ResNet-50 pré-entraîné ImageNet SANS fine-tuning contrastif.
Objectif : Mesurer l'apport réel de l'entraînement contrastif (Exp #2).

Comparaison :
- Baseline (ce script) : ResNet-50 ImageNet pur
- Exp #2 : ResNet-50 ImageNet + 23 epochs contrastif

Conditions identiques :
✓ Même architecture (ResNet-50)
✓ Même pré-entraînement (ImageNet)
✓ Même données (train/test avec oversampling)
✓ Même classifieur (SVM RBF)
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torchvision import models

# Imports locaux
import config
from dataset import VRDDataset

def extract_imagenet_features_with_random_projection(dataloader, device):
    """
    Extrait les features avec ARCHITECTURE IDENTIQUE à Exp #2, 
    mais projection head NON ENTRAÎNÉE (poids random).
    
    Architecture :
    - ResNet-50 ImageNet (gelé) - IDENTIQUE à Exp #2
    - Projection 4096→128 (RANDOM, pas entraînée) - Différence avec Exp #2
    
    Returns:
        X: Embeddings 128D
        y: Labels
    """
    # 1. Charger ResNet-50 pré-entraîné ImageNet (IDENTIQUE à Exp #2)
    print("\nChargement de l'architecture complète (ResNet-50 + Projection)...")
    print("  - ResNet-50 : Poids ImageNet (gelés)")
    print("  - Projection : Poids RANDOM (NON entraînés)")
    
    backbone = models.resnet50(weights='DEFAULT').to(device)
    backbone.fc = torch.nn.Identity()
    
    # Geler le backbone (comme dans Exp #2)
    for param in backbone.parameters():
        param.requires_grad = False
    
    # 2. Créer projection head IDENTIQUE à Exp #2 (mais poids RANDOM)
    projection = torch.nn.Sequential(
        torch.nn.Linear(2048 * 2, 1024),  # Identique à Exp #2
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 128)
    ).to(device)
    
    # Poids sont initialisés random (pas de load_state_dict)
    projection.eval()
    
    print(f"✓ Architecture chargée (MÊME structure que Exp #2, projection RANDOM)")
    
    features = []
    labels_list = []
    
    print(f"\nExtraction des embeddings 128D (via projection random)...")
    with torch.no_grad():
        for img_s, img_o, _, label in tqdm(dataloader):
            img_s = img_s.to(device)
            img_o = img_o.to(device)
            
            # Extraire features ResNet-50
            feat_s = backbone(img_s)  # [batch, 2048]
            feat_o = backbone(img_o)  # [batch, 2048]
            
            # Concaténer
            feat_concat = torch.cat([feat_s, feat_o], dim=1)  # [batch, 4096]
            
            # Passer par projection RANDOM
            embeddings = projection(feat_concat)  # [batch, 128]
            
            features.append(embeddings.cpu().numpy())
            labels_list.append(label.numpy())
    
    X = np.vstack(features)
    y = np.concatenate(labels_list)
    return X, y

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Proportion'})
    plt.ylabel('Vraie Relation', fontsize=12)
    plt.xlabel('Relation Prédite', fontsize=12)
    plt.title('Matrice de Confusion - ResNet-50 ImageNet Baseline (SANS Contrastif)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Matrice de confusion sauvegardée : {save_path}")
    plt.close()

def main():
    print("=" * 70)
    print("ÉVALUATION BASELINE - Projection RANDOM vs ENTRAÎNÉE (Contrastif)")
    print("=" * 70)
    print("\n🎯 COMPARAISON JUSTE :")
    print("   - Baseline (ce test) : ResNet-50 ImageNet + Projection RANDOM")
    print("   - Exp #2             : ResNet-50 ImageNet + Projection ENTRAÎNÉE (23 epochs)")
    print("\n⚠️  SEULE DIFFÉRENCE : Poids de la projection (random vs optimisés)")
    print("   → Permet d'isoler l'apport RÉEL du fine-tuning contrastif")
    
    # Créer dossier de résultats
    baseline_dir = os.path.join(config.BASE_DIR, "baseline_random_projection")
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Chargement des données (IDENTIQUE à Exp #2)
    print(f"\n{'='*70}")
    print(f"CHARGEMENT DES DONNÉES (Identique à Exp #2)")
    print(f"{'='*70}")
    
    train_ds = VRDDataset(subset='train')
    test_ds = VRDDataset(subset='test')
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n✅ Train: {len(train_ds)} échantillons (avec oversampling)")
    print(f"✅ Test: {len(test_ds)} échantillons")
    
    # Extraction des embeddings avec projection RANDOM
    print(f"\n{'='*70}")
    print(f"EXTRACTION DES EMBEDDINGS (Architecture identique à Exp #2)")
    print(f"{'='*70}")
    
    X_train, y_train = extract_imagenet_features_with_random_projection(train_loader, config.device)
    X_test, y_test = extract_imagenet_features_with_random_projection(test_loader, config.device)
    
    print(f"\n📊 Dimensions des embeddings :")
    print(f"  - Train : {X_train.shape} (128D via projection RANDOM)")
    print(f"  - Test  : {X_test.shape} (128D via projection RANDOM)")
    
    # Classification avec SVM (IDENTIQUE à Exp #2)
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION (SVM - Identique à Exp #2)")
    print(f"{'='*70}")
    
    print(f"\nEntraînement SVM (RBF kernel, C=1.0)...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    
    print(f"Prédiction sur test set...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Résultats
    print(f"\n{'='*70}")
    print(f"RÉSULTATS - BASELINE (Projection RANDOM)")
    print(f"{'='*70}")
    
    print(f"\n🎯 Accuracy sur Test Set : {accuracy*100:.2f}%")
    print(f"\n📊 COMPARAISON JUSTE :")
    print(f"  - Baseline (projection RANDOM)     : {accuracy*100:.2f}%")
    print(f"  - Exp #2 (projection ENTRAÎNÉE)    : 62.40%")
    
    delta = 62.40 - (accuracy * 100)
    if delta > 0:
        print(f"\n✅ APPORT DU FINE-TUNING CONTRASTIF : +{delta:.2f}%")
        print(f"   → Les 23 epochs d'entraînement ont AMÉLIORÉ le modèle")
    else:
        print(f"\n⚠️  Le contrastif n'a PAS amélioré : {delta:.2f}%")
        print(f"   → L'entraînement n'a pas apporté de gain")
    
    # Rapport détaillé
    target_names = list(test_ds.rel2idx.keys())
    
    print(f"\n{'='*70}")
    print(f"RAPPORT DÉTAILLÉ PAR RELATION")
    print(f"{'='*70}\n")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Matrice de confusion
    cm_path = os.path.join(baseline_dir, "confusion_matrix_baseline.png")
    plot_confusion_matrix(y_test, y_pred, target_names, cm_path)
    
    # Sauvegarder rapport
    report_path = os.path.join(baseline_dir, "baseline_results.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BASELINE - Projection RANDOM vs ENTRAÎNÉE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Configuration :\n")
        f.write(f"  - Architecture : ResNet-50 (gelé) + Projection 4096→128\n")
        f.write(f"  - ResNet-50 : Poids ImageNet pré-entraînés (gelés)\n")
        f.write(f"  - Projection : Poids RANDOM (NON entraînés)\n")
        f.write(f"  - Embeddings : 128D\n")
        f.write(f"  - Classifieur : SVM (RBF, C=1.0)\n\n")
        f.write(f"Dataset :\n")
        f.write(f"  - Train : {len(train_ds)} échantillons\n")
        f.write(f"  - Test  : {len(test_ds)} échantillons\n\n")
        f.write(f"Accuracy : {accuracy*100:.2f}%\n\n")
        f.write(f"Comparaison (Architecture Identique) :\n")
        f.write(f"  - Baseline (projection RANDOM)    : {accuracy*100:.2f}%\n")
        f.write(f"  - Exp #2 (projection ENTRAÎNÉE)   : 62.40%\n")
        f.write(f"  - Apport entraînement contrastif  : {delta:+.2f}%\n\n")
        f.write("=" * 70 + "\n")
        f.write("RAPPORT DÉTAILLÉ\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    print(f"\n📝 Résultats sauvegardés : {report_path}")
    print(f"📁 Tous les fichiers dans : {baseline_dir}")
    
    print(f"\n{'='*70}")
    print(f"✅ ÉVALUATION BASELINE TERMINÉE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
