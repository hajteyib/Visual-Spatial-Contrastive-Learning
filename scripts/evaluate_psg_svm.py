#!/usr/bin/env python3
"""
Évaluation PSG avec SVM sur embeddings VRD.
Transfer learning : VRD features → SVM classifier → PSG labels

Pipeline :
1. Extraire embeddings du modèle VRD (visual + spatial)
2. Split PSG en train/test
3. Entraîner SVM sur PSG train embeddings
4. Évaluer sur PSG test
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
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import VisualEncoder, SpatialEncoder
from src.utils.geometry import get_spatial_vector
from src import config

# Config
PSG_JSON = "psg/psg_val_test.json"
PSG_IMG_DIR = "psg/val2017"
CHECKPOINT_DIR = "checkpoints/exp_20251202_175017"
EMBEDDINGS_FILE = "psg_embeddings.pkl"
SVM_MODEL_FILE = "psg_svm_model.pkl"

def load_vrd_model(checkpoint_dir, device):
    """Charger modèle VRD pour extraction features"""
    print(f"Loading VRD model from {checkpoint_dir}...")
    
    visual_path = os.path.join(checkpoint_dir, "best_visual_encoder.pth")
    spatial_path = os.path.join(checkpoint_dir, "best_spatial_encoder.pth")
    
    # Détecter architecture
    visual_checkpoint = torch.load(visual_path, map_location='cpu')
    proj_shape = visual_checkpoint['projection.0.weight'].shape
    
    if proj_shape[1] == 4096:
        print("✅ ResNet-50 detected")
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
    
    print("✅ Model loaded")
    return visual_model, spatial_model

def extract_embeddings(visual_model, spatial_model, psg_data, img_dir, device):
    """Extraire embeddings pour toutes les relations PSG"""
    print("\n=== Extracting Embeddings ===\n")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    labels = []
    errors = []
    
    with torch.no_grad():
        for item in tqdm(psg_data, desc="Processing images"):
            img_file = os.path.basename(item['file_name'])
            img_path = os.path.join(img_dir, img_file)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                errors.append(f"Image load error: {e}")
                continue
            
            img_w, img_h = image.size
            annotations = item['annotations']  # BBoxes ici !
            
            for rel in item['relations']:
                try:
                    subj_idx, obj_idx, pred_idx = rel
                    
                    # PSG : bbox format [x1, y1, x2, y2] (bbox_mode=0)
                    subj_bbox = annotations[subj_idx]['bbox']
                    obj_bbox = annotations[obj_idx]['bbox']
                    
                    # Crops
                    crop_s = image.crop(subj_bbox)
                    crop_o = image.crop(obj_bbox)
                    
                    tensor_s = transform(crop_s).unsqueeze(0).to(device)
                    tensor_o = transform(crop_o).unsqueeze(0).to(device)
                    
                    # Spatial vector
                    spatial_vec = get_spatial_vector(subj_bbox, obj_bbox, img_w, img_h)
                    spatial_vec = torch.tensor(spatial_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Embeddings VRD
                    z_v = visual_model(tensor_s, tensor_o)
                    z_s = spatial_model(spatial_vec)
                    
                    # Fusion : concaténer visual + spatial
                    embedding = torch.cat([z_v, z_s], dim=1).cpu().numpy()[0]
                    
                    embeddings.append(embedding)
                    labels.append(pred_idx)
                    
                except Exception as e:
                    if len(errors) < 5:  # Limite errors
                        errors.append(f"Relation error: {e}")
                    continue
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"✅ Extracted {len(embeddings)} embeddings")
    if len(errors) > 0:
        print(f"⚠️  {len(errors)} errors (showing first 5):")
        for err in errors[:5]:
            print(f"   - {err}")
    
    print(f"   Shape: {embeddings.shape}")
    print(f"   Unique classes: {len(np.unique(labels))}")
    
    return embeddings, labels

def train_svm(embeddings, labels):
    """Entraîner SVM sur embeddings"""
    print("\n=== Training SVM ===\n")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # SVM avec RBF kernel
    print("\nTraining SVM (this may take a while)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=True)
    svm.fit(X_train, y_train)
    
    print("✅ SVM trained")
    
    # Évaluation
    print("\n=== Evaluation ===\n")
    y_pred = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Rapport détaillé
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return svm, accuracy, X_test, y_test, y_pred

def main():
    print("="*70)
    print("PSG EVALUATION WITH SVM (Transfer Learning)")
    print("="*70)
    print("\nPipeline:")
    print("1. Load VRD model (62.40% on VRD)")
    print("2. Extract embeddings for PSG relations")
    print("3. Train SVM on PSG embeddings")
    print("4. Evaluate SVM accuracy\n")
    
    device = config.device
    print(f"Device: {device}\n")
    
    # 1. Load VRD model
    visual_model, spatial_model = load_vrd_model(CHECKPOINT_DIR, device)
    
    # 2. Load PSG data
    print(f"\nLoading PSG data from {PSG_JSON}...")
    with open(PSG_JSON, 'r') as f:
        psg_dataset = json.load(f)
    
    # Filtrer images disponibles
    available_data = []
    for item in psg_dataset['data']:
        img_file = os.path.basename(item['file_name'])
        if os.path.exists(os.path.join(PSG_IMG_DIR, img_file)):
            available_data.append(item)
    
    print(f"✅ {len(available_data)} images available")
    predicate_classes = psg_dataset['predicate_classes']
    print(f"   {len(predicate_classes)} PSG predicate classes")
    
    # 3. Extract embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"\nLoading cached embeddings from {EMBEDDINGS_FILE}...")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings, labels = pickle.load(f)
        print(f"✅ Loaded {len(embeddings)} embeddings")
    else:
        embeddings, labels = extract_embeddings(
            visual_model, spatial_model, available_data, PSG_IMG_DIR, device
        )
        
        # Sauvegarder
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump((embeddings, labels), f)
        print(f"✅ Saved embeddings to {EMBEDDINGS_FILE}")
    
    # 4. Train SVM
    svm, accuracy, X_test, y_test, y_pred = train_svm(embeddings, labels)
    
    # Sauvegarder SVM
    with open(SVM_MODEL_FILE, 'wb') as f:
        pickle.dump(svm, f)
    print(f"\n✅ SVM saved to {SVM_MODEL_FILE}")
    
    # Résumé final
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nVRD Model: ResNet-50 (62.40% on VRD)")
    print(f"PSG Relations: {len(embeddings)} total")
    print(f"SVM Test Accuracy: {accuracy*100:.2f}%")
    print(f"\n✅ Cross-dataset generalization: VRD → PSG")

if __name__ == "__main__":
    main()
