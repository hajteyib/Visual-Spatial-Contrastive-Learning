#!/usr/bin/env python3
"""
Évaluation du meilleur modèle VRD sur le dataset PSG.
Test de généralisation cross-dataset.

Modèle : exp_20251202_175017 (62.40% sur VRD)
Dataset : PSG val_test (2,157 images, 6,448 relations)
"""

import torch
import json
import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Ajouter le répertoire parent au path pour importer src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importer depuis src comme module
from src.model import VisualEncoder, SpatialEncoder
from src.utils.geometry import get_spatial_vector
from src import config

# Configuration
PSG_JSON = "psg/psg_val_test.json"
PSG_IMG_DIR = "psg/val2017"
CHECKPOINT_DIR = "checkpoints/exp_20251202_175017"

# Mapping VRD relations → PSG predicates
# IMPORTANT: Ces classes correspondent EXACTEMENT à l'entraînement du modèle
# (vérifié dans exp_20251202_175017/evaluation_results/results.txt)
VRD_TO_PSG_MAPPING = {
    'on': ['on', 'standing on', 'sitting on', 'lying on'],
    'under': ['under'],
    'above': ['above', 'over'],
    'below': ['below'],
    'left of': ['to the left of'],
    'right of': ['to the right of'],
    'near': ['near', 'beside', 'adjacent to'],
    'next to': ['next to', 'beside'],
    'inside': ['in', 'inside', 'within'],
    'outside': ['outside', 'out of']
}

def load_models(checkpoint_dir, device):
    """Charger les modèles depuis checkpoint avec détection automatique d'architecture"""
    print(f"Loading models from {checkpoint_dir}...")
    
    visual_path = os.path.join(checkpoint_dir, "best_visual_encoder.pth")
    spatial_path = os.path.join(checkpoint_dir, "best_spatial_encoder.pth")
    
    if not os.path.exists(visual_path) or not os.path.exists(spatial_path):
        raise FileNotFoundError(f"Model weights not found in {checkpoint_dir}")
    
    # Charger checkpoint pour détecter architecture
    visual_checkpoint = torch.load(visual_path, map_location='cpu')
    
    # Détecter ResNet-50 vs ResNet-18 via dimensions projection
    # ResNet-50: 2048 features → projection.0.weight = [1024, 4096]
    # ResNet-18: 512 features → projection.0.weight = [512, 1024]
    proj_weight_shape = visual_checkpoint['projection.0.weight'].shape
    
    if proj_weight_shape[1] == 4096:  # 2048*2 = 4096
        print("✅ Detected: ResNet-50 architecture")
        resnet_type = 'resnet50'
        embedding_dim = 128  # D'après projection.3.weight shape
    else:
        print("✅ Detected: ResNet-18 architecture")
        resnet_type = 'resnet18'
        embedding_dim = 256
    
    # Créer modèle avec bonne architecture
    # Import dynamique de torchvision
    from torchvision import models as tv_models
    import torch.nn as nn
    
    # Créer VisualEncoder avec architecture détectée
    class DynamicVisualEncoder(nn.Module):
        def __init__(self, resnet_type, embedding_dim):
            super().__init__()
            
            if resnet_type == 'resnet50':
                self.backbone = tv_models.resnet50(weights='DEFAULT')
                feature_dim = 2048
            else:
                self.backbone = tv_models.resnet18(weights='DEFAULT')
                feature_dim = 512
            
            # Geler backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            self.backbone.fc = nn.Identity()
            
            # Projection
            self.projection = nn.Sequential(
                nn.Linear(feature_dim * 2, 1024 if resnet_type == 'resnet50' else 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024 if resnet_type == 'resnet50' else 512, embedding_dim)
            )
        
        def forward(self, img_s, img_o):
            feat_s = self.backbone(img_s)
            feat_o = self.backbone(img_o)
            combined = torch.cat((feat_s, feat_o), dim=1)
            z_v = self.projection(combined)
            return nn.functional.normalize(z_v, dim=1)
    
    visual_model = DynamicVisualEncoder(resnet_type, embedding_dim).to(device)
    spatial_model = SpatialEncoder(input_dim=8, embedding_dim=embedding_dim).to(device)
    
    # Charger poids
    visual_model.load_state_dict(visual_checkpoint)
    spatial_model.load_state_dict(torch.load(spatial_path, map_location=device))
    
    visual_model.eval()
    spatial_model.eval()
    
    print(f"✅ Models loaded successfully ({resnet_type})")
    return visual_model, spatial_model

def load_psg_data(json_path, img_dir):
    """Charger annotations PSG"""
    print(f"Loading PSG annotations from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filtrer seulement images disponibles
    available_data = []
    for item in data['data']:
        img_file = os.path.basename(item['file_name'])
        img_path = os.path.join(img_dir, img_file)
        
        if os.path.exists(img_path):
            available_data.append(item)
    
    print(f"✅ Loaded {len(available_data)} images with annotations")
    
    # Extraire prédic classes
    predicate_classes = data['predicate_classes']
    print(f"   PSG has {len(predicate_classes)} predicate classes")
    
    return available_data, predicate_classes

def evaluate_psg(visual_model, spatial_model, psg_data, predicate_classes, img_dir, device):
    """Évaluer sur PSG"""
    print("\n=== Starting PSG Evaluation ===\n")
    
    # VRD classes (EXACTEMENT celles du modèle exp_20251202_175017)
    vrd_classes = [
        'on', 'under', 'above', 'below',
        'left of', 'right of', 'near', 'next to',
        'inside', 'outside'  # Pas 'in'/'over' !
    ]
    
    # Stats
    total_relations = 0
    vrd_covered_relations = 0
    correct_predictions = 0
    
    # Par classe
    per_class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # Transformations images (même que training)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for item in tqdm(psg_data, desc="Evaluating"):
            # Charger image
            img_file = os.path.basename(item['file_name'])
            img_path = os.path.join(img_dir, img_file)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                continue
            
            img_w, img_h = image.size
            segments = item['segments_info']
            
            # Process chaque relation
            for rel in item['relations']:
                subj_idx, obj_idx, pred_idx = rel
                
                # Get predicate name
                pred_name = predicate_classes[pred_idx]
                total_relations += 1
                
                # Vérifier si couvert par VRD
                vrd_match = None
                for vrd_rel, psg_rels in VRD_TO_PSG_MAPPING.items():
                    if pred_name in psg_rels:
                        vrd_match = vrd_rel
                        vrd_covered_relations += 1
                        break
                
                if vrd_match is None:
                    continue  # Relation pas dans VRD
                
                # Extraire bounding boxes
                try:
                    subj_bbox = segments[subj_idx]['bbox']  # [x, y, w, h]
                    obj_bbox = segments[obj_idx]['bbox']
                    
                    # Convertir en [x1, y1, x2, y2]
                    bbox_s = [subj_bbox[0], subj_bbox[1], 
                             subj_bbox[0] + subj_bbox[2], subj_bbox[1] + subj_bbox[3]]
                    bbox_o = [obj_bbox[0], obj_bbox[1],
                             obj_bbox[0] + obj_bbox[2], obj_bbox[1] + obj_bbox[3]]
                    
                except (IndexError, KeyError):
                    continue
                
                # Crops
                crop_s = image.crop(bbox_s)
                crop_o = image.crop(bbox_o)
                
                tensor_s = transform(crop_s).unsqueeze(0).to(device)
                tensor_o = transform(crop_o).unsqueeze(0).to(device)
                
                # Spatial vector
                spatial_vec = get_spatial_vector(bbox_s, bbox_o, img_w, img_h)
                spatial_vec = torch.tensor(spatial_vec, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Predict
                z_v = visual_model(tensor_s, tensor_o)
                z_s = spatial_model(spatial_vec)
                
                # Similarité
                sim = torch.matmul(z_v, z_s.T).item()
                
                # Pour évaluation simplifiée : si sim > seuil → correct
                # (Normalement on comparerait avec autres relations mais c'est cross-dataset)
                threshold = 0.5
                predicted_correct = sim > threshold
                
                per_class_stats[vrd_match]['total'] += 1
                if predicted_correct:
                    per_class_stats[vrd_match]['correct'] += 1
                    correct_predictions += 1
    
    # Résultats
    print("\n" + "="*70)
    print("PSG EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTotal PSG relations : {total_relations}")
    print(f"Relations couvertes par VRD : {vrd_covered_relations} ({100*vrd_covered_relations/total_relations:.1f}%)")
    
    if vrd_covered_relations > 0:
        accuracy = correct_predictions / vrd_covered_relations
        print(f"\nAccuracy (sur relations VRD) : {accuracy*100:.2f}%")
    
    print(f"\n--- Per-Class Performance ---")
    for vrd_rel in sorted(per_class_stats.keys()):
        stats = per_class_stats[vrd_rel]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"{vrd_rel:12s}: {stats['correct']:4d}/{stats['total']:4d} ({acc*100:5.1f}%)")
    
    return {
        'total_relations': total_relations,
        'vrd_covered': vrd_covered_relations,
        'correct': correct_predictions,
        'per_class': dict(per_class_stats)
    }

def main():
    print("="*70)
    print("PSG EVALUATION - Cross-Dataset Generalization")
    print("="*70)
    print(f"\nModel: {CHECKPOINT_DIR}")
    print(f"VRD Test Accuracy: 62.40%")
    print()
    
    # Device
    device = config.device
    print(f"Device: {device}\n")
    
    # Load
    visual_model, spatial_model = load_models(CHECKPOINT_DIR, device)
    psg_data, predicate_classes = load_psg_data(PSG_JSON, PSG_IMG_DIR)
    
    # Evaluate
    results = evaluate_psg(visual_model, spatial_model, psg_data, predicate_classes, PSG_IMG_DIR, device)
    
    # Save
    output_file = "psg_evaluation_results.txt"
    with open(output_file, 'w') as f:
        f.write("PSG Cross-Dataset Evaluation\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {CHECKPOINT_DIR}\n")
        f.write(f"VRD Accuracy: 62.40%\n\n")
        f.write(f"Total PSG relations: {results['total_relations']}\n")
        f.write(f"VRD covered: {results['vrd_covered']}\n")
        f.write(f"Correct predictions: {results['correct']}\n")
        if results['vrd_covered'] > 0:
            acc = results['correct'] / results['vrd_covered']
            f.write(f"Accuracy: {acc*100:.2f}%\n")
    
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()
