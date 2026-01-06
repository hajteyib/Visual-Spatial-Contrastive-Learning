import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

# --- CONFIGURATION ---
# Mets le chemin vers ton dossier d'images et ton fichier JSON
IMG_DIR = "vrd/sg_train_images" # Ou "vrd/train_images" selon ton dossier
JSON_PATH = "vrd/sg_train_annotations.json" 

def visualize_scene_graph():
    print(f"Chargement du JSON depuis {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # On prend une image au hasard
    entry = random.choice(data)
    
    filename = entry.get('filename', 'Inconnu')
    img_path = os.path.join(IMG_DIR, filename)
    
    print(f"--- Image sélectionnée : {filename} ---")
    
    # Vérifier si l'image existe
    if not os.path.exists(img_path):
        print(f"❌ Image introuvable ici : {img_path}")
        print("Vérifie la variable IMG_DIR dans le script.")
        return

    # Charger l'image
    image = Image.open(img_path)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Récupérer les listes
    objects_list = entry['objects']
    relationships = entry['relationships']
    
    # DEBUG : Afficher la structure d'un objet pour être sûr des BBox
    if len(objects_list) > 0:
        print("Exemple de structure d'un objet (pour vérifier x,y,w,h) :")
        print(objects_list[0])

    # Dessiner les relations spatiales
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, rel in enumerate(relationships):
        predicat = rel['relationship']
        
        # On filtre un peu pour ne pas surcharger (on cherche les relations spatiales)
        if predicat not in ['on', 'under', 'left of', 'right of', 'above', 'below', 'near']:
            continue
            
        # Récupérer les indices
        sujet_idx = rel['objects'][0]
        objet_idx = rel['objects'][1]
        
        # Récupérer les infos des objets
        sujet_info = objects_list[sujet_idx]
        objet_info = objects_list[objet_idx]
        
        # --- DESSINER SUJET ---
        # ADAPTATION : Souvent dans ce format c'est 'x', 'y', 'w', 'h'
        # Si ça plante ici, regarde le print de debug plus haut
        bbox_s = sujet_info['bbox']
        sx, sy, sw, sh = bbox_s['x'], bbox_s['y'], bbox_s['w'], bbox_s['h']
        name_s = sujet_info['names'][0] if 'names' in sujet_info else 'sujet'
        
        rect_s = patches.Rectangle((sx, sy), sw, sh, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_s)
        ax.text(sx, sy, name_s, color='white', backgroundcolor='red', fontsize=8)

        # --- DESSINER OBJET ---
        bbox_o = objet_info['bbox']
        ox, oy, ow, oh = bbox_o['x'], bbox_o['y'], bbox_o['w'], bbox_o['h']
        name_o = objet_info['names'][0] if 'names' in objet_info else 'objet'
        
        rect_o = patches.Rectangle((ox, oy), ow, oh, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect_o)
        ax.text(ox, oy, name_o, color='white', backgroundcolor='blue', fontsize=8)
        
        print(f"Relation : {name_s} --[{predicat}]--> {name_o}")
        
        # On arrête après 3 relations pour ne pas faire un dessin illisible
        if i > 2: break

    plt.title(f"Relations Spatiales : {filename}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    try:
        visualize_scene_graph()
    except KeyError as e:
        print(f"\n❌ Erreur de clé JSON : {e}")
        print("Le script cherche des clés 'x', 'y', 'w', 'h'. Regarde l'exemple imprimé ci-dessus pour corriger.")
    except Exception as e:
        print(f"Erreur : {e}")