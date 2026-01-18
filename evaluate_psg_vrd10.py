#!/usr/bin/env python3
"""
Évaluation PSG avec SVM - SEULEMENT les 10 classes VRD spatiales.

Pipeline:
1. Filtrer PSG pour garder seulement relations spatiales VRD (10 classes)
2. Extraire embeddings VRD pour ces relations
3. Entraîner SVM sur 10 classes
4. Évaluer accuracy

Mapping VRD → PSG (relations spatiales uniquement):
  on, under, above, below, left of, right of, near, next to, inside, outside
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import VisualEncoder, SpatialEncoder
from src.utils.geometry import get_spatial_vector
from src import config

# Config
PSG_JSON = "psg/psg_val_test.json"
PSG_IMG_DIR = "psg/val2017"
CHECKPOINT_DIR = "checkpoints/exp_20251202_175017"

# Mapping VRD (10 classes) → PSG predicates
VRD_TO_PSG = {
    'on': ['on', 'standing on', 'sitting on', 'lying on'],
    'under': ['under'],
    'above': ['above', 'over'],
    'below': ['below'],
    'left of': ['to the left of'],
    'right of': ['to the right of'],
    'near': ['near', 'beside', 'adjacent to'],
    'next to': ['next to'],
    'inside': ['in', 'inside', 'within'],
    'outside': ['outside']
}

VRD_CLASSES = list(VRD_TO_PSG.keys())

def load_vrd_model(checkpoint_dir, device):
    """Charger modèle VRD"""
    print(f"Loading VRD model from {checkpoint_dir}...")
    
    visual_path = os.path.join(checkpoint_dir, "best_visual_encoder.pth")
    spatial_path = os.path.join(checkpoint_dir, "best_spatial_encoder.pth")
    
    visual_checkpoint = torch.load(visual_path, map_location='cpu')
    proj_shape = visual_checkpoint['projection.0.weight'].shape
    
    if proj_shape[1] == 4096:
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
    else:
        visual_model = VisualEncoder(256).to(device)
        spatial_model = SpatialEncoder(8, 256).to(device)
    
    visual_model.load_state_dict(visual_checkpoint)
    spatial_model.load_state_dict(torch.load(spatial_path, map_location=device))
    
    visual_model.eval()
    spatial_model.eval()
    
    print("✅ Model loaded (ResNet-50)")
    return visual_model, spatial_model

def filter_psg_spatial(psg_data, predicate_classes):
    """Filtrer PSG pour garder seulement relations spatiales VRD"""
    print("\n=== Filtering PSG for VRD Spatial Relations ===\n")
    
    # Créer mapping PSG predicate → VRD class
    psg_to_vrd = {}
    for vrd_class, psg_preds in VRD_TO_PSG.items():
        for psg_pred in psg_preds:
            if psg_pred in predicate_classes:
                psg_idx = predicate_classes.index(psg_pred)
                psg_to_vrd[psg_idx] = vrd_class
    
    print(f"PSG predicates mapped to VRD: {len(psg_to_vrd)}")
    for psg_idx, vrd_class in psg_to_vrd.items():
        print(f"  {predicate_classes[psg_idx]} → {vrd_class}")
    
    # Filtrer relations
    filtered_data = []
    total_relations = 0
    kept_relations = 0
    
    for item in psg_data:
        new_relations = []
        for rel in item['relations']:
            total_relations += 1
            pred_idx = rel[2]
            if pred_idx in psg_to_vrd:
                new_relations.append(rel)
                kept_relations += 1
        
        if new_relations:
            item_copy = item.copy()
            item_copy['relations'] = new_relations
            filtered_data.append(item_copy)
    
    print(f"\n✅ Filtered: {kept_relations}/{total_relations} relations ({100*kept_relations/total_relations:.1f}%)")
    print(f"✅ {len(filtered_data)} images with spatial relations")
    
    return filtered_data, psg_to_vrd

def extract_embeddings(visual_model, spatial_model, psg_data, psg_to_vrd, img_dir, device):
    """Extraire embeddings pour relations spatiales"""
    print("\n=== Extracting Embeddings ===\n")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    labels = []
    
    # Mapping VRD class → index
    vrd_to_idx = {cls: i for i, cls in enumerate(VRD_CLASSES)}
    
    with torch.no_grad():
        for item in tqdm(psg_data, desc="Processing"):
            img_file = os.path.basename(item['file_name'])
            img_path = os.path.join(img_dir, img_file)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                continue
            
            img_w, img_h = image.size
            annotations = item['annotations']
            
            for rel in item['relations']:
                try:
                    subj_idx, obj_idx, pred_idx = rel
                    
                    # Convertir PSG pred → VRD class
                    vrd_class = psg_to_vrd[pred_idx]
                    vrd_label = vrd_to_idx[vrd_class]
                    
                    # Bboxes
                    subj_bbox = annotations[subj_idx]['bbox']
                    obj_bbox = annotations[obj_idx]['bbox']
                    
                    # Crops
                    crop_s = image.crop(subj_bbox)
                    crop_o = image.crop(obj_bbox)
                    
                    tensor_s = transform(crop_s).unsqueeze(0).to(device)
                    tensor_o = transform(crop_o).unsqueeze(0).to(device)
                    
                    # Spatial
                    spatial_vec = get_spatial_vector(subj_bbox, obj_bbox, img_w, img_h)
                    spatial_vec = torch.tensor(spatial_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Embeddings
                    z_v = visual_model(tensor_s, tensor_o)
                    z_s = spatial_model(spatial_vec)
                    
                    # Fusion
                    embedding = torch.cat([z_v, z_s], dim=1).cpu().numpy()[0]
                    
                    embeddings.append(embedding)
                    labels.append(vrd_label)
                    
                except Exception as e:
                    continue
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"✅ Extracted {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape}")
    
    # Distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nClass distribution:")
    for vrd_idx, count in sorted(label_counts.items()):
        print(f"  {VRD_CLASSES[vrd_idx]:12s}: {count:4d} ({100*count/len(labels):5.1f}%)")
    
    return embeddings, labels

def train_svm(embeddings, labels):
    """Entraîner SVM sur 10 classes VRD"""
    print("\n=== Training SVM (10 VRD classes) ===\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    
    print("✅ SVM trained")
    
    # Évaluation
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%\n")
    
    # Classes présentes dans le test set
    unique_labels = np.unique(y_test)
    present_classes = [VRD_CLASSES[i] for i in unique_labels]
    
    # Rapport par classe
    print(classification_report(y_test, y_pred, 
                               target_names=present_classes,
                               labels=unique_labels,
                               zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print("\nConfusion Matrix:")
    print(f"Classes: {present_classes}")
    print(cm)
    
    return svm, accuracy

def main():
    print("="*70)
    print("PSG EVALUATION - VRD 10 SPATIAL CLASSES ONLY")
    print("="*70)
    print("\nVRD Classes:", VRD_CLASSES)
    print()
    
    device = config.device
    print(f"Device: {device}\n")
    
    # Load model
    visual_model, spatial_model = load_vrd_model(CHECKPOINT_DIR, device)
    
    # Load PSG
    with open(PSG_JSON, 'r') as f:
        psg_dataset = json.load(f)
    
    available_data = []
    for item in psg_dataset['data']:
        img_file = os.path.basename(item['file_name'])
        if os.path.exists(os.path.join(PSG_IMG_DIR, img_file)):
            available_data.append(item)
    
    predicate_classes = psg_dataset['predicate_classes']
    
    # Filter pour spatial relations
    filtered_data, psg_to_vrd = filter_psg_spatial(available_data, predicate_classes)
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(
        visual_model, spatial_model, filtered_data, psg_to_vrd, PSG_IMG_DIR, device
    )
    
    # Train SVM
    svm, accuracy = train_svm(embeddings, labels)
    
    # Save
    with open('psg_vrd10_svm.pkl', 'wb') as f:
        pickle.dump(svm, f)
    print(f"\n✅ SVM saved to psg_vrd10_svm.pkl")
    
    print(f"\n{'='*70}")
    print(f"FINAL: VRD→PSG (10 spatial classes): {accuracy*100:.2f}%")
    print('='*70)

if __name__ == "__main__":
    main()
