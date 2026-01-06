import torch
import os

# --- 1. CONFIGURATION MATÉRIEL ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Configuration chargée. Device utilisé : {device}")

# --- 2. HYPERPARAMÈTRES ---
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
EPOCHS = 30  # Exp #4: Supervised contrastive needs more epochs
IMG_SIZE = 128
EMBEDDING_DIM = 256

# --- 3. CHEMINS (CORRECTION ICI) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modifie cette ligne si nécessaire :
# Cela doit pointer vers le dossier qui CONTIENT les fichiers json
DATA_DIR = os.path.join(BASE_DIR, 'vrd') 

# Pour le debug, on imprime où on cherche
print(f"DEBUG: Le dossier DATA_DIR est configuré sur : {DATA_DIR}")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'sg_train_images')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'sg_test_images')
TRAIN_JSON = os.path.join(DATA_DIR, 'sg_train_annotations.json')
TEST_JSON = os.path.join(DATA_DIR, 'sg_test_annotations.json')

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)