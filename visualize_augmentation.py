"""
Script pour visualiser les augmentations de données appliquées aux images.
Montre l'image originale + 3 versions augmentées côte à côte.
"""

import matplotlib.pyplot as plt
from PIL import Image
import os
import config
from dataset import VRDDataset
import torchvision.transforms as transforms

def visualize_augmentations(num_examples=3):
    """
    Affiche des exemples d'augmentations de données.
    
    Args:
        num_examples: Nombre d'images différentes à visualiser
    """
    
    # Dataset avec augmentation
    train_ds = VRDDataset(subset='train')
    
    # Transformation SANS augmentation (pour comparaison)
    transform_no_aug = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Transformation AVEC augmentation (même que dans le dataset)
    transform_with_aug = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    print(f"\n📸 Visualisation des augmentations sur {num_examples} images\n")
    
    for example_idx in range(num_examples):
        # Récupérer une image du dataset
        sample = train_ds.samples[example_idx * 100]  # Espacer les exemples
        img_path = os.path.join(train_ds.img_dir, sample['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Extraire les crops comme dans le dataset
            def get_bbox_list(obj_info):
                b = obj_info['bbox']
                x, y, w, h = b['x'], b['y'], b['w'], b['h']
                return [x, y, x + w, y + h]
            
            bbox_s = get_bbox_list(sample['sujet_info'])
            crop_s = image.crop(bbox_s)
            
            # Version SANS augmentation
            img_original = transform_no_aug(crop_s).permute(1, 2, 0).numpy()
            
            # 3 versions AVEC augmentation
            img_aug1 = transform_with_aug(crop_s).permute(1, 2, 0).numpy()
            img_aug2 = transform_with_aug(crop_s).permute(1, 2, 0).numpy()
            img_aug3 = transform_with_aug(crop_s).permute(1, 2, 0).numpy()
            
            # Affichage
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(img_original)
            axes[0].set_title('Original (Sans Augmentation)', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(img_aug1)
            axes[1].set_title('Augmentation #1', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(img_aug2)
            axes[2].set_title('Augmentation #2', fontsize=12)
            axes[2].axis('off')
            
            axes[3].imshow(img_aug3)
            axes[3].set_title('Augmentation #3', fontsize=12)
            axes[3].axis('off')
            
            relation = sample['predicat']
            plt.suptitle(f"Exemple {example_idx + 1} - Relation: '{relation}' | {sample['filename']}", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Sauvegarder
            save_path = os.path.join(config.BASE_DIR, f"augmentation_example_{example_idx + 1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Exemple {example_idx + 1} sauvegardé : {save_path}")
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"❌ Erreur sur l'exemple {example_idx + 1}: {e}")
            continue
    
    print(f"\n🎨 Visualisation terminée !")
    print(f"\nAugmentations appliquées :")
    print(f"  1. RandomHorizontalFlip (p=0.5) - Symétrie horizontale aléatoire")
    print(f"  2. ColorJitter (brightness=0.2, contrast=0.2) - Variations de luminosité/contraste")
    print(f"\n💡 Note : Les augmentations sont ALÉATOIRES, donc chaque passage donne un résultat différent.")

if __name__ == "__main__":
    visualize_augmentations(num_examples=3)
