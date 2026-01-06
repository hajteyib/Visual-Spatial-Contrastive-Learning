import torch
import math

def get_spatial_vector(bbox_sujet, bbox_objet, img_w, img_h):
    """
    Calcule un descripteur géométrique (Spatial Vector) entre deux objets.
    bbox format : [x1, y1, x2, y2]
    """
    
    # 1. Calcul des centres (cx, cy) normalisés (0 à 1)
    s_cx = ((bbox_sujet[0] + bbox_sujet[2]) / 2) / img_w
    s_cy = ((bbox_sujet[1] + bbox_sujet[3]) / 2) / img_h
    
    o_cx = ((bbox_objet[0] + bbox_objet[2]) / 2) / img_w
    o_cy = ((bbox_objet[1] + bbox_objet[3]) / 2) / img_h
    
    # 2. Calcul des aires (Normalisées)
    s_w = bbox_sujet[2] - bbox_sujet[0]
    s_h = bbox_sujet[3] - bbox_sujet[1]
    s_area = (s_w * s_h) / (img_w * img_h)
    
    o_w = bbox_objet[2] - bbox_objet[0]
    o_h = bbox_objet[3] - bbox_objet[1]
    o_area = (o_w * o_h) / (img_w * img_h)
    
    # 3. Features relatives
    # Delta X et Delta Y
    dx = o_cx - s_cx
    dy = o_cy - s_cy
    
    # Distance euclidienne au carré
    dist_sq = dx**2 + dy**2
    
    # Angle (très important pour gauche/droite/haut/bas)
    angle_rad = math.atan2(dy, dx)
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    
    # Ratio des tailles
    size_ratio = s_area / (o_area + 1e-6)

    # 4. Vecteur final (taille 8)
    spatial_vector = torch.tensor([
        dx, 
        dy, 
        dist_sq, 
        sin_a, 
        cos_a, 
        s_area, 
        o_area, 
        size_ratio
    ], dtype=torch.float32)
    
    return spatial_vector