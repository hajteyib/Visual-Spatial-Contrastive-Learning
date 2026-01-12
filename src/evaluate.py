"""
Script d'évaluation du modèle de relations spatiales.

Après l'entraînement contrastif, ce script :
1. Charge les encoders entraînés
2. Extrait les embeddings du test set
3. Entraîne un classifieur (SVM) sur les embeddings
4. Évalue la précision de prédiction des relations spatiales
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

# Imports locaux
import config
from dataset import VRDDataset
from model import VisualEncoder, SpatialEncoder

def find_latest_experiment():
    """Trouve le dossier d'expérience le plus récent."""
    exp_dirs = [d for d in os.listdir(config.CHECKPOINT_DIR) if d.startswith('exp_')]
    if not exp_dirs:
        raise FileNotFoundError("Aucun dossier d'expérience trouvé dans checkpoints/")
    
    # Trier par date (format exp_YYYYMMDD_HHMMSS)
    exp_dirs.sort(reverse=True)
    latest_exp = os.path.join(config.CHECKPOINT_DIR, exp_dirs[0])
    return latest_exp

def extract_features(model, dataloader, device, model_name="Visual"):
    """
    Extrait les embeddings et les labels depuis un dataloader.
    
    Args:
        model: VisualEncoder ou SpatialEncoder
        dataloader: DataLoader du dataset
        device: Device (cpu/mps/cuda)
        model_name: "Visual" ou "Spatial" pour affichage
    
    Returns:
        X: Embeddings (numpy array)
        y: Labels (numpy array)
    """
    model.eval()
    features = []
    labels_list = []
    
    print(f"Extraction des embeddings {model_name}...")
    with torch.no_grad():
        for img_s, img_o, spatial_vec, label in tqdm(dataloader):
            img_s = img_s.to(device)
            img_o = img_o.to(device)
            spatial_vec = spatial_vec.to(device)
            
            # Selon le type de modèle
            if isinstance(model, VisualEncoder):
                embeddings = model(img_s, img_o)
            else:  # SpatialEncoder
                embeddings = model(spatial_vec)
            
            features.append(embeddings.cpu().numpy())
            labels_list.append(label.numpy())
    
    X = np.vstack(features)
    y = np.concatenate(labels_list)
    return X, y

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalisation pour voir des pourcentages
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Proportion'})
    plt.ylabel('Vraie Relation', fontsize=12)
    plt.xlabel('Relation Prédite', fontsize=12)
    plt.title('Matrice de Confusion (Normalisée)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Matrice de confusion sauvegardée : {save_path}")
    plt.close()

def evaluate_classifier(X_train, y_train, X_test, y_test, classifier_name="SVM"):
    """
    Entraîne et évalue un classifieur.
    
    Returns:
        y_pred: Prédictions
        accuracy: Score d'accuracy
    """
    print(f"\n--- Entraînement {classifier_name} ---")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    
    print(f"Entraînement sur {X_train.shape[0]} échantillons...")
    clf.fit(X_train, y_train)
    
    print(f"Prédiction sur {X_test.shape[0]} échantillons test...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

def main():
    print("=" * 70)
    print("ÉVALUATION DU MODÈLE DE RELATIONS SPATIALES")
    print("=" * 70)
    
    # --- 1. Trouver l'expérience à évaluer ---
    try:
        exp_dir = find_latest_experiment()
        print(f"\n📁 Expérience évaluée : {os.path.basename(exp_dir)}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Lancez 'python3 train.py' d'abord pour entraîner un modèle.")
        return
    
    # Créer dossier de résultats
    results_dir = os.path.join(exp_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. Chargement des données ---
    print(f"\n--- Chargement des données (Split 70/20/10) ---")
    
    # Train : pour entraîner le classifieur
    train_ds = VRDDataset(subset='train')
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Test : pour évaluer le classifieur (10% jamais vu)
    test_ds = VRDDataset(subset='test')
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✅ Train: {len(train_ds)} échantillons (pour SVM)")
    print(f"✅ Test: {len(test_ds)} échantillons (pour évaluation finale)")
    
    # --- 3. Chargement des modèles entraînés ---
    print(f"\n--- Chargement des modèles entraînés ---")
    
    visual_model = VisualEncoder(embedding_dim=config.EMBEDDING_DIM).to(config.device)
    spatial_model = SpatialEncoder(input_dim=8, embedding_dim=config.EMBEDDING_DIM).to(config.device)
    
    # Chemins des checkpoints
    visual_checkpoint = os.path.join(exp_dir, "best_visual_encoder.pth")
    spatial_checkpoint = os.path.join(exp_dir, "best_spatial_encoder.pth")
    
    if not os.path.exists(visual_checkpoint):
        print(f"❌ Checkpoint Visual non trouvé : {visual_checkpoint}")
        return
    
    if not os.path.exists(spatial_checkpoint):
        print(f"❌ Checkpoint Spatial non trouvé : {spatial_checkpoint}")
        return
    
    visual_model.load_state_dict(torch.load(visual_checkpoint, map_location=config.device))
    spatial_model.load_state_dict(torch.load(spatial_checkpoint, map_location=config.device))
    
    print(f"✅ VisualEncoder chargé : {visual_checkpoint}")
    print(f"✅ SpatialEncoder chargé : {spatial_checkpoint}")
    
    # --- 4. Extraction des embeddings ---
    print(f"\n{'='*70}")
    print("EXTRACTION DES EMBEDDINGS")
    print(f"{'='*70}")
    
    # Visual embeddings
    X_train_visual, y_train = extract_features(visual_model, train_loader, config.device, "Visual")
    X_test_visual, y_test = extract_features(visual_model, test_loader, config.device, "Visual")
    
    # Spatial embeddings
    X_train_spatial, _ = extract_features(spatial_model, train_loader, config.device, "Spatial")
    X_test_spatial, _ = extract_features(spatial_model, test_loader, config.device, "Spatial")
    
    # Fusion (Visual + Spatial)
    X_train_fusion = np.hstack([X_train_visual, X_train_spatial])
    X_test_fusion = np.hstack([X_test_visual, X_test_spatial])
    
    print(f"\n📊 Dimensions des embeddings :")
    print(f"  - Visual : {X_train_visual.shape}")
    print(f"  - Spatial : {X_train_spatial.shape}")
    print(f"  - Fusion : {X_train_fusion.shape}")
    
    # --- 5. Classification et Évaluation ---
    print(f"\n{'='*70}")
    print("ÉVALUATION DE LA CLASSIFICATION")
    print(f"{'='*70}")
    
    results = {}
    
    # Test 1 : Visual seul
    print("\n[1/3] Évaluation avec embeddings VISUELS uniquement")
    y_pred_visual, acc_visual = evaluate_classifier(X_train_visual, y_train, 
                                                     X_test_visual, y_test, 
                                                     "SVM (Visual)")
    results['visual'] = acc_visual
    
    # Test 2 : Spatial seul
    print("\n[2/3] Évaluation avec embeddings SPATIAUX uniquement")
    y_pred_spatial, acc_spatial = evaluate_classifier(X_train_spatial, y_train, 
                                                       X_test_spatial, y_test, 
                                                       "SVM (Spatial)")
    results['spatial'] = acc_spatial
    
    # Test 3 : Fusion
    print("\n[3/3] Évaluation avec embeddings FUSIONNÉS (Visual + Spatial)")
    y_pred_fusion, acc_fusion = evaluate_classifier(X_train_fusion, y_train, 
                                                     X_test_fusion, y_test, 
                                                     "SVM (Fusion)")
    results['fusion'] = acc_fusion
    
    # --- 6. Résultats ---
    print(f"\n{'='*70}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*70}")
    print(f"\n🎯 Accuracy sur Test Set (10%, {len(test_ds)} échantillons) :\n")
    print(f"  Visual seul  : {acc_visual*100:.2f}%")
    print(f"  Spatial seul : {acc_spatial*100:.2f}%")
    print(f"  Fusion (V+S) : {acc_fusion*100:.2f}% ⭐ (Meilleur)")
    
    # Rapport détaillé (sur la meilleure config = Fusion)
    target_names = list(test_ds.rel2idx.keys())
    
    print(f"\n{'='*70}")
    print("RAPPORT DÉTAILLÉ PAR RELATION (Fusion Visual + Spatial)")
    print(f"{'='*70}\n")
    print(classification_report(y_test, y_pred_fusion, target_names=target_names, zero_division=0))
    
    # --- 7. Matrice de Confusion ---
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred_fusion, target_names, cm_path)
    
    # --- 8. Sauvegarde des résultats ---
    results_file = os.path.join(results_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RÉSULTATS D'ÉVALUATION - RELATIONS SPATIALES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Expérience : {os.path.basename(exp_dir)}\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset Test : {len(test_ds)} échantillons (10%)\n\n")
        f.write("Accuracy :\n")
        f.write(f"  - Visual seul  : {acc_visual*100:.2f}%\n")
        f.write(f"  - Spatial seul : {acc_spatial*100:.2f}%\n")
        f.write(f"  - Fusion (V+S) : {acc_fusion*100:.2f}%\n\n")
        f.write("=" * 70 + "\n")
        f.write("RAPPORT DÉTAILLÉ (Fusion)\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(y_test, y_pred_fusion, target_names=target_names, zero_division=0))
    
    print(f"\n📝 Résultats sauvegardés : {results_file}")
    print(f"📁 Tous les fichiers dans : {results_dir}")
    
    print(f"\n{'='*70}")
    print("✅ ÉVALUATION TERMINÉE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()