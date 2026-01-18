#!/usr/bin/env python3
"""
Script pour télécharger seulement les images PSG nécessaires depuis COCO.
Utilise psg_val_test.json pour télécharger ~2,177 images test (~200 MB).
"""

import json
import os
from urllib.request import urlopen
from urllib.error import URLError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PSG_JSON = "psg/psg_val_test.json"
OUTPUT_DIR = "psg/val2017"
COCO_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{filename}"
MAX_WORKERS = 8  # Téléchargements parallèles

def download_image(filename, output_dir):
    """Télécharger une image COCO"""
    url = COCO_URL_TEMPLATE.format(filename=os.path.basename(filename))
    output_path = os.path.join(output_dir, os.path.basename(filename))
    
    # Skip si déjà téléchargé
    if os.path.exists(output_path):
        return True, filename
    
    try:
        with urlopen(url, timeout=10) as response:
            data = response.read()
        
        with open(output_path, 'wb') as f:
            f.write(data)
        
        return True, filename
    except Exception as e:
        return False, f"{filename}: {e}"

def main():
    print("=== PSG Selective Image Downloader ===\n")
    
    # 1. Charger annotations PSG
    print(f"Loading {PSG_JSON}...")
    with open(PSG_JSON, 'r') as f:
        psg_data = json.load(f)
    
    # 2. Extraire noms de fichiers
    image_files = [item['file_name'] for item in psg_data['data']]
    print(f"Found {len(image_files)} images to download\n")
    
    # 3. Créer dossier output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4. Télécharger en parallèle
    print(f"Downloading to {OUTPUT_DIR}...")
    print(f"Using {MAX_WORKERS} parallel workers\n")
    
    success_count = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, fname, OUTPUT_DIR): fname 
            for fname in image_files
        }
        
        with tqdm(total=len(image_files), desc="Downloading") as pbar:
            for future in as_completed(futures):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    failed.append(result)
                pbar.update(1)
    
    # 5. Résumé
    print(f"\n=== Download Complete ===")
    print(f"✅ Success: {success_count}/{len(image_files)}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print("\nFailed files:")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")
    
    # Estimer taille
    if success_count > 0:
        sample_size = sum(
            os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
            for f in os.listdir(OUTPUT_DIR)[:100]
        ) / 100  # Moyenne sur 100 images
        
        total_size_mb = (sample_size * success_count) / 1024 / 1024
        print(f"\nEstimated total size: ~{total_size_mb:.0f} MB")

if __name__ == "__main__":
    main()
