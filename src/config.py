import torch
import os

# --- 1. CONFIGURATION MATÉRIEL ---
# Auto-detect best device: CUDA (Colab/GPU) > MPS (Mac) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Configuration chargée. Device utilisé : {device}")

# --- 2. HYPERPARAMÈTRES ---
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
EPOCHS = 50  # Exp #5: Long training (was 30)
IMG_SIZE = 128
EMBEDDING_DIM = 256

# --- 3. CHEMINS ---
# config.py est dans src/, donc on remonte d'un niveau pour trouver vrd/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'vrd') 

# Pour le debug
print(f"DEBUG: Le dossier DATA_DIR est configuré sur : {DATA_DIR}")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'sg_train_images')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'sg_test_images')
TRAIN_JSON = os.path.join(DATA_DIR, 'sg_train_annotations.json')
TEST_JSON = os.path.join(DATA_DIR, 'sg_test_annotations.json')

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)