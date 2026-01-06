import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import torchvision.transforms as transforms
from utils_geometry import get_spatial_vector
import config

class VRDDataset(Dataset):
    def __init__(self, subset='train', use_original_split=False):
        """
        subset: 'train', 'val', ou 'test'
        use_original_split: Si True, utilise l'ancien split train/test des JSONs.
                           Si False (défaut), crée un split 70/20/10 depuis le train JSON.
        """
        self.subset = subset
        
        # 1. Définition des chemins selon le subset
        if use_original_split:
            # Ancien comportement : train vs test selon les JSONs séparés
            if subset == 'train':
                self.img_dir = config.TRAIN_IMG_DIR
                self.json_path = config.TRAIN_JSON
            else:
                self.img_dir = config.TEST_IMG_DIR
                self.json_path = config.TEST_JSON
        else:
            # Nouveau comportement : tout depuis train JSON, split après
            self.img_dir = config.TRAIN_IMG_DIR
            self.json_path = config.TRAIN_JSON

        # 2. Liste des relations spatiales qu'on veut garder (Filtre)
        # ✅ SÉLECTION FINALE (10 classes équilibrées - Accord avec utilisateur)
        # Retirées : inside (181), outside (54) - trop rares
        # Retirées : behind, in front of - mal représentées par vecteur 8D
        # Ajoutées : in (3186), over (1504) - bien représentables
        self.target_relations = [
            'on', 'under', 'above', 'below', 
            'left of', 'right of', 'near', 'next to',
            'in', 'over'  # ✅ CORRECT
        ]
        
        # Création d'un dictionnaire pour convertir les mots en chiffres (Label encoding)
        # ex: {'on': 0, 'under': 1, ...}
        self.rel2idx = {rel: i for i, rel in enumerate(self.target_relations)}
        
        # 3. Chargement et Filtrage du JSON
        print(f"Chargement du dataset {subset}...")
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)

        self.samples = []
        
        # On parcourt chaque image
        for entry in raw_data:
            filename = entry.get('filename', None)
            if not filename: continue
            
            objects = entry['objects']
            relationships = entry['relationships']
            
            # On parcourt chaque relation dans l'image
            for rel in relationships:
                predicat = rel['relationship'].lower().strip() # Nettoyage du texte
                
                # Si c'est une relation spatiale connue
                if predicat in self.target_relations:
                    
                    # Récupération des indices
                    idx_sujet = rel['objects'][0]
                    idx_objet = rel['objects'][1]
                    
                    # On stocke juste ce qu'il faut pour charger plus tard
                    self.samples.append({
                        'filename': filename,
                        'predicat': predicat,
                        'sujet_info': objects[idx_sujet],
                        'objet_info': objects[idx_objet]
                    })
        
        # 4. SPLIT TRAIN/VAL/TEST (70/20/10) avec seed fixe
        if not use_original_split:
            import random
            random.seed(42)  # Reproductibilité
            random.shuffle(self.samples)
            
            total = len(self.samples)
            train_size = int(0.7 * total)
            val_size = int(0.2 * total)
            
            if subset == 'train':
                self.samples = self.samples[:train_size]
            elif subset == 'val':
                self.samples = self.samples[train_size:train_size + val_size]
            elif subset == 'test':
                self.samples = self.samples[train_size + val_size:]
            else:
                raise ValueError(f"Subset '{subset}' inconnu. Utilisez 'train', 'val' ou 'test'.")

        print(f"Dataset {subset} prêt : {len(self.samples)} paires spatiales trouvées.")

        # 5. Transformateur d'images (avec data augmentation pour train uniquement)
        if subset == 'train':
            # TRAIN : avec augmentation (flip + variations couleur)
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal aléatoire
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variations lumière
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # VAL/TEST : sans augmentation
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # --- A. Chargement Image ---
        img_path = os.path.join(self.img_dir, sample['filename'])
        
        try:
            # .convert('RGB') gère les images noir et blanc qui feraient planter le code
            image = Image.open(img_path).convert('RGB')
            img_w, img_h = image.size
        except Exception as e:
            # En cas d'image corrompue, on renvoie l'item suivant (astuce robuste)
            return self.__getitem__((idx + 1) % len(self))

        # --- B. Extraction des Bounding Boxes ---
        # Le JSON donne {bbox: {x, y, w, h}}
        # On veut [x1, y1, x2, y2]
        
        def get_bbox_list(obj_info):
            # Correction ici basée sur ton retour d'erreur précédent
            b = obj_info['bbox'] 
            x, y, w, h = b['x'], b['y'], b['w'], b['h']
            return [x, y, x + w, y + h]

        bbox_s = get_bbox_list(sample['sujet_info'])
        bbox_o = get_bbox_list(sample['objet_info'])

        # --- C. Crops (Découpage) ---
        # image.crop attend (left, top, right, bottom)
        crop_s = image.crop(bbox_s)
        crop_o = image.crop(bbox_o)
        
        # Transformation en Tenseurs PyTorch
        tensor_s = self.transform(crop_s)
        tensor_o = self.transform(crop_o)

        # --- D. Vecteur Géométrique (Méthode 1) ---
        spatial_vec = get_spatial_vector(bbox_s, bbox_o, img_w, img_h)

        # --- E. Label ---
        label_id = self.rel2idx[sample['predicat']]
        
        return tensor_s, tensor_o, spatial_vec, torch.tensor(label_id)

# --- Zone de Test ---
if __name__ == "__main__":
    # Petit script pour vérifier que tout charge bien
    try:
        ds = VRDDataset(subset='train')
        print(f"Test lecture item 0 :")
        s_img, o_img, s_vec, label = ds[0]
        print(f"Shape Image Sujet : {s_img.shape}") # Doit être [3, 128, 128]
        print(f"Vecteur Spatial : {s_vec}")         # Doit être un tenseur de taille 8
        print(f"Label ID : {label}")
        print("✅ Dataset fonctionnel !")
    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("Vérifie que tes dossiers 'data/train_images' et le json sont au bon endroit définie dans config.py")