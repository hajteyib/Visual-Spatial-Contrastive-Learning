import torch
import torch.nn as nn
from torchvision import models
from . import config

class VisualEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(VisualEncoder, self).__init__()
        
        # === RESNET-18 (Exp #4) ===
        # Retour à ResNet-18 (prouvé efficace : 61.67% en Exp #1)
        # Plus robuste qu'EfficientNet pour notre tâche
        self.backbone = models.resnet18(weights='DEFAULT')
        
        # 🔒 GELER LE BACKBONE (transfer learning)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # ResNet-18 features: 512D (vs 1280D EfficientNet-B0)
        feature_dim = 512
        
        # On enlève la dernière couche (classification)
        self.backbone.fc = nn.Identity()
        
        # === PROJECTION HEAD ===
        # Sujet (512) + Objet (512) = 1024D
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # 1024 → 512
            nn.ReLU(),
            nn.Dropout(0.4),  # Augmenté de 0.3 à 0.4 (Exp #4)
            nn.Linear(512, embedding_dim)  # 512 → 256
        )

    def forward(self, img_sujet, img_objet):
        # Passage dans ResNet-18
        feat_s = self.backbone(img_sujet)  # [Batch, 512]
        feat_o = self.backbone(img_objet)  # [Batch, 512]
        
        # Fusion des informations visuelles
        combined = torch.cat((feat_s, feat_o), dim=1)  # [Batch, 1024]
        
        # Projection dans l'espace d'embedding
        z_v = self.projection(combined)  # [Batch, 256]
        
        # Normalisation L2 (Important pour loss contrastive)
        z_v = nn.functional.normalize(z_v, dim=1)
        return z_v

class SpatialEncoder(nn.Module):
    def __init__(self, input_dim=8, embedding_dim=256):
        super(SpatialEncoder, self).__init__()
        
        # MLP pour comprendre la géométrie
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Augmenté de 0.2 à 0.3 (Exp #4)
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Augmenté de 0.2 à 0.3 (Exp #4)
            nn.Linear(128, embedding_dim)  # Sortie 256D
        )

    def forward(self, spatial_vector):
        z_s = self.net(spatial_vector)
        # Normalisation L2
        z_s = nn.functional.normalize(z_s, dim=1)
        return z_s

# --- Test unitaire ---
if __name__ == "__main__":
    # Test rapide pour voir si les dimensions collent
    device = torch.device("cpu")
    v_enc = VisualEncoder()
    s_enc = SpatialEncoder()
    
    dummy_img = torch.randn(4, 3, 128, 128) # Batch de 4 images
    dummy_vec = torch.randn(4, 8)           # Batch de 4 vecteurs géo
    
    zv = v_enc(dummy_img, dummy_img)
    zs = s_enc(dummy_vec)
    
    print(f"Sortie Visuelle : {zv.shape}") # Doit être [4, 128]
    print(f"Sortie Spatiale : {zs.shape}") # Doit être [4, 128]
    print("✅ Modèle OK")