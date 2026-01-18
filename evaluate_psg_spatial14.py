#!/usr/bin/env python3
"""
Évaluation PSG - 14 CLASSES SPATIALES
Transfer learning VRD → PSG spatial relations

Classes PSG spatiales (14):
  on, over, beside, in front of, in, attached to, hanging from,
  on back of, standing on, lying on, sitting on, walking on,
  parked on, driving on, leaning on
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

# 14 classes spatiales PSG (indices vérifiés)
SPATIAL_PRED_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 47, 48, 55]

SPATIAL_CLASSES = [
    'over', 'in front of', 'beside', 'on', 'in', 
    'attached to', 'hanging from', 'on back of',
    'standing on', 'lying on', 'sitting on',
    'parked on', 'driving on', 'leaning on'
]

def load_vrd_model(checkpoint_dir, device):
    """Charger modèle VRD"""
    print(f"Loading VRD model from {checkpoint_dir}...")
    
    visual_path = os.path.join(checkpoint_dir, "best_visual_encoder.pth")
    spatial_path = os.path.join(checkpoint_dir, "best_spatial_encoder.pth")
    
    visual_checkpoint = torch.load(visual_path, map_location='cpu')
    
    # Détection ResNet-50
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
    
    print("✅ Model loaded (ResNet-50)")
    return visual_model, spatial_model

def filter_psg_spatial(psg_data):
    """Filtrer pour 14 classes spatiales"""
    print("\n=== Filtering for 14 Spatial Classes ===\n")
    
    spatial_set = set(SPATIAL_PRED_INDICES)
    
    filtered_data = []
    total_relations = 0
    kept_relations = 0
    
    for item in psg_data:
        new_relations = []
        for rel in item['relations']:
            total_relations += 1
            pred_idx = rel[2]
            if pred_idx in spatial_set:
                new_relations.append(rel)
                kept_relations += 1
        
        if new_relations:
            item_copy = item.copy()
            item_copy['relations'] = new_relations
            filtered_data.append(item_copy)
    
    print(f"✅ Kept: {kept_relations}/{total_relations} relations ({100*kept_relations/total_relations:.1f}%)")
    print(f"✅ {len(filtered_data)} images with spatial relations")
    
    return filtered_data

def extract_embeddings(visual_model, spatial_model, psg_data, img_dir, device):
    """Extraire embeddings"""
    print("\n=== Extracting Embeddings ===\n")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    labels = []
    
    # Mapping PSG pred_idx → class_idx (0-13)
    pred_to_class = {pred_idx: i for i, pred_idx in enumerate(SPATIAL_PRED_INDICES)}
    
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
                    spatial_vec = torch.from_numpy(np.array(spatial_vec, dtype=np.float32)).unsqueeze(0).to(device)
                    
                    # Embeddings
                    z_v = visual_model(tensor_s, tensor_o)
                    z_s = spatial_model(spatial_vec)
                    
                    # Fusion
                    embedding = torch.cat([z_v, z_s], dim=1).cpu().numpy()[0]
                    
                    embeddings.append(embedding)
                    labels.append(pred_to_class[pred_idx])
                    
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
    for class_idx in range(len(SPATIAL_CLASSES)):
        count = label_counts.get(class_idx, 0)
        if count > 0:
            pct = 100 * count / len(labels)
            print(f"  {SPATIAL_CLASSES[class_idx]:15s} : {count:4d} ({pct:5.1f}%)")
    
    return embeddings, labels

def train_svm(embeddings, labels):
    """Entraîner SVM sur 14 classes"""
    print("\n=== Training SVM (14 spatial classes) ===\n")
    
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
    
    # Classes présentes
    unique_labels = np.unique(y_test)
    present_classes = [SPATIAL_CLASSES[i] for i in unique_labels]
    
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
    print("PSG EVALUATION - 14 SPATIAL CLASSES")
    print("="*70)
    print(f"\nSpatial classes: {SPATIAL_CLASSES}")
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
    
    print(f"✅ {len(available_data)} PSG images available")
    
    # Filter spatial
    filtered_data = filter_psg_spatial(available_data)
    
    # Extract
    embeddings, labels = extract_embeddings(
        visual_model, spatial_model, filtered_data, PSG_IMG_DIR, device
    )
    
    # Train SVM
    svm, accuracy = train_svm(embeddings, labels)
    
    # Save
    with open('psg_spatial14_svm.pkl', 'wb') as f:
        pickle.dump(svm, f)
    print(f"\n✅ SVM saved to psg_spatial14_svm.pkl")
    
    print(f"\n{'='*70}")
    print(f"FINAL: VRD→PSG (14 spatial classes): {accuracy*100:.2f}%")
    print('='*70)

if __name__ == "__main__":
    main()
