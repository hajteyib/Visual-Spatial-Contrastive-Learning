#!/usr/bin/env python3
"""
PSG EVALUATION - TOUTES LES 56 CLASSES
Transfer Learning: VRD features → SVM → PSG 56 classes

Métriques calculées:
- Accuracy (top-1)
- Recall@1, @5, @10
- Per-class metrics
- Confusion matrix

Résultats sauvegardés dans: psg_results_56classes/
"""

import torch
import json
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import VisualEncoder, SpatialEncoder
from src.utils.geometry import get_spatial_vector
from src import config

# Config
PSG_JSON = "psg/psg_val_test.json"
PSG_IMG_DIR = "psg/val2017"
CHECKPOINT_DIR = "checkpoints/exp_20251202_175017"
RESULTS_DIR = "psg_results_56classes"

def create_results_dir():
    """Créer répertoire de résultats"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"✅ Results directory: {RESULTS_DIR}/")

def load_vrd_model(checkpoint_dir, device):
    """Charger modèle VRD ResNet-50"""
    print(f"\nLoading VRD model from {checkpoint_dir}...")
    
    visual_path = os.path.join(checkpoint_dir, "best_visual_encoder.pth")
    spatial_path = os.path.join(checkpoint_dir, "best_spatial_encoder.pth")
    
    visual_checkpoint = torch.load(visual_path, map_location='cpu')
    
    from torchvision import models as tv_models
    import torch.nn as nn
    
    class DynamicVisualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = tv_models.resnet50(weights='DEFAULT')
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.fc = nn.Identity()
            self.projection = nn.Sequential(
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 128)
            )
        
        def forward(self, img_s, img_o):
            feat_s = self.backbone(img_s)
            feat_o = self.backbone(img_o)
            combined = torch.cat((feat_s, feat_o), dim=1)
            z_v = self.projection(combined)
            return nn.functional.normalize(z_v, dim=1)
    
    visual_model = DynamicVisualEncoder().to(device)
    spatial_model = SpatialEncoder(8, 128).to(device)
    
    visual_model.load_state_dict(visual_checkpoint)
    spatial_model.load_state_dict(torch.load(spatial_path, map_location=device))
    
    visual_model.eval()
    spatial_model.eval()
    
    print("✅ VRD Model loaded (ResNet-50)")
    return visual_model, spatial_model

def extract_embeddings(visual_model, spatial_model, psg_data, img_dir, device):
    """Extraire embeddings pour TOUTES les relations PSG"""
    print("\n=== Extracting Embeddings (56 classes) ===\n")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    labels = []
    errors = 0
    
    with torch.no_grad():
        for item in tqdm(psg_data, desc="Processing images"):
            img_file = os.path.basename(item['file_name'])
            img_path = os.path.join(img_dir, img_file)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                errors += 1
                continue
            
            img_w, img_h = image.size
            annotations = item['annotations']
            
            for rel in item['relations']:
                try:
                    subj_idx, obj_idx, pred_idx = rel
                    
                    # Bboxes
                    subj_bbox = annotations[subj_idx]['bbox']
                    obj_bbox = annotations[obj_idx]['bbox']
                    
                    # Crops
                    crop_s = image.crop(subj_bbox)
                    crop_o = image.crop(obj_bbox)
                    
                    tensor_s = transform(crop_s).unsqueeze(0).to(device)
                    tensor_o = transform(crop_o).unsqueeze(0).to(device)
                    
                    # Spatial vector
                    spatial_vec = get_spatial_vector(subj_bbox, obj_bbox, img_w, img_h)
                    spatial_vec = torch.from_numpy(np.array(spatial_vec, dtype=np.float32)).unsqueeze(0).to(device)
                    
                    # Embeddings VRD
                    z_v = visual_model(tensor_s, tensor_o)
                    z_s = spatial_model(spatial_vec)
                    
                    # Fusion
                    embedding = torch.cat([z_v, z_s], dim=1).cpu().numpy()[0]
                    
                    embeddings.append(embedding)
                    labels.append(pred_idx)
                    
                except Exception as e:
                    errors += 1
                    continue
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"\n✅ Extracted {len(embeddings)} embeddings")
    print(f"   Errors: {errors}")
    print(f"   Shape: {embeddings.shape}")
    
    return embeddings, labels

def calculate_recall_at_k(y_true, decision_scores, k_values=[1, 5, 10]):
    """Calculer Recall@K pour comparaison avec retrieval"""
    results = {}
    
    for k in k_values:
        # Pour chaque sample, prendre les top-k prédictions
        top_k_preds = np.argsort(decision_scores, axis=1)[:, -k:]  # Top-K indices
        
        # Vérifier si la vraie classe est dans le top-K
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        recall_k = correct.mean()
        
        results[f'Recall@{k}'] = recall_k
    
    return results

def train_and_evaluate_svm(embeddings, labels, predicate_classes):
    """Entraîner SVM sur 56 classes avec Recall@K"""
    print("\n=== Training SVM (56 PSG classes) ===\n")
    
    # Split (sans stratify car certaines classes n'ont qu'1 sample)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Classes: {len(np.unique(labels))}")
    
    # Train SVM
    print("\nTraining SVM (this may take a while)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr')
    svm.fit(X_train, y_train)
    
    print("✅ SVM trained")
    
    # Predictions
    print("\n=== Evaluation ===\n")
    y_pred = svm.predict(X_test)
    decision_scores = svm.decision_function(X_test)
    
    # Accuracy (top-1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (top-1): {accuracy*100:.2f}%")
    
    # Recall@K
    recall_metrics = calculate_recall_at_k(y_test, decision_scores, k_values=[1, 5, 10])
    print(f"\nRecall@1:  {recall_metrics['Recall@1']*100:.2f}%")
    print(f"Recall@5:  {recall_metrics['Recall@5']*100:.2f}%")
    print(f"Recall@10: {recall_metrics['Recall@10']*100:.2f}%")
    
    # Per-class report
    print("\nGenerating classification report...")
    
    # Filtrer pour classes présentes dans test
    unique_test_labels = np.unique(y_test)
    present_classes = [predicate_classes[i] for i in unique_test_labels]
    
    report = classification_report(y_test, y_pred,
                                   labels=unique_test_labels,
                                   target_names=present_classes,
                                   zero_division=0,
                                   output_dict=True)
    
    report_str = classification_report(y_test, y_pred,
                                      labels=unique_test_labels,
                                      target_names=present_classes,
                                      zero_division=0)
    
    # Confusion matrix (trop grande pour afficher, on la sauvegarde)
    cm = confusion_matrix(y_test, y_pred, labels=unique_test_labels)
    
    return svm, accuracy, recall_metrics, report, report_str, cm, unique_test_labels, y_test, y_pred

def save_results(accuracy, recall_metrics, report, report_str, cm, unique_test_labels, 
                predicate_classes, label_counts, y_test, y_pred):
    """Sauvegarder tous les résultats"""
    print(f"\n=== Saving Results to {RESULTS_DIR}/ ===\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Résumé principal
    summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PSG EVALUATION - 56 CLASSES COMPLÈTES\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: VRD ResNet-50 + SVM\n")
        f.write(f"Dataset: PSG val_test (6,448 relations)\n\n")
        
        f.write("="*70 + "\n")
        f.write("RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy (top-1): {accuracy*100:.2f}%\n\n")
        f.write(f"Recall@1:  {recall_metrics['Recall@1']*100:.2f}%\n")
        f.write(f"Recall@5:  {recall_metrics['Recall@5']*100:.2f}%\n")
        f.write(f"Recall@10: {recall_metrics['Recall@10']*100:.2f}%\n\n")
        
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report_str)
        
        f.write("\n" + "="*70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("="*70 + "\n\n")
        for pred_idx, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / sum(label_counts.values())
            f.write(f"{pred_idx:2d}. {predicate_classes[pred_idx]:20s} : {count:4d} ({pct:5.1f}%)\n")
    
    print(f"✅ Summary: {summary_file}")
    
    # 2. Metrics détaillés (JSON)
    metrics_file = os.path.join(RESULTS_DIR, "metrics.json")
    metrics_data = {
        'accuracy': float(accuracy),
        'recall_at_1': float(recall_metrics['Recall@1']),
        'recall_at_5': float(recall_metrics['Recall@5']),
        'recall_at_10': float(recall_metrics['Recall@10']),
        'num_classes': len(unique_test_labels),
        'test_samples': len(y_test),
        'timestamp': timestamp
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✅ Metrics (JSON): {metrics_file}")
    
    # 3. Confusion matrix (numpy)
    cm_file = os.path.join(RESULTS_DIR, "confusion_matrix.npy")
    np.save(cm_file, cm)
    print(f"✅ Confusion matrix: {cm_file}")
    
    # 4. Class mapping
    mapping_file = os.path.join(RESULTS_DIR, "class_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("PSG Class Mapping\n")
        f.write("="*50 + "\n\n")
        for idx in unique_test_labels:
            f.write(f"{idx:2d}: {predicate_classes[idx]}\n")
    
    print(f"✅ Class mapping: {mapping_file}")

def main():
    print("="*70)
    print("PSG EVALUATION - 56 CLASSES COMPLÈTES")
    print("="*70)
    print("\nTransfer Learning: VRD features → SVM → PSG 56 classes")
    print(f"Device: {config.device}\n")
    
    # Créer répertoire résultats
    create_results_dir()
    
    # Load VRD model
    visual_model, spatial_model = load_vrd_model(CHECKPOINT_DIR, config.device)
    
    # Load PSG data
    print(f"\nLoading PSG data from {PSG_JSON}...")
    with open(PSG_JSON, 'r') as f:
        psg_dataset = json.load(f)
    
    predicate_classes = psg_dataset['predicate_classes']
    print(f"✅ PSG has {len(predicate_classes)} predicate classes")
    
    # Filter available images
    available_data = []
    for item in psg_dataset['data']:
        img_file = os.path.basename(item['file_name'])
        if os.path.exists(os.path.join(PSG_IMG_DIR, img_file)):
            available_data.append(item)
    
    print(f"✅ {len(available_data)} images available")
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(
        visual_model, spatial_model, available_data, PSG_IMG_DIR, config.device
    )
    
    # Class distribution
    label_counts = Counter(labels)
    print(f"\nUnique classes in data: {len(label_counts)}")
    
    # Train SVM
    svm, accuracy, recall_metrics, report, report_str, cm, unique_test_labels, y_test, y_pred = train_and_evaluate_svm(
        embeddings, labels, predicate_classes
    )
    
    # Save results
    save_results(accuracy, recall_metrics, report, report_str, cm, unique_test_labels,
                predicate_classes, label_counts, y_test, y_pred)
    
    # Save SVM model
    svm_file = os.path.join(RESULTS_DIR, "svm_model.pkl")
    with open(svm_file, 'wb') as f:
        pickle.dump(svm, f)
    print(f"✅ SVM model: {svm_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved in: {RESULTS_DIR}/")
    print(f"\n🎯 Final Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Recall@1: {recall_metrics['Recall@1']*100:.2f}%")
    print(f"📊 Recall@5: {recall_metrics['Recall@5']*100:.2f}%")
    print(f"📊 Recall@10: {recall_metrics['Recall@10']*100:.2f}%")

if __name__ == "__main__":
    main()
