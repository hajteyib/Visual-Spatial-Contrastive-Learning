# EXP #5 - COLAB TRAINING NOTEBOOK

## What This Does

This notebook:
1. Clones your GitHub repo
2. Mounts Google Drive (for VRD dataset)
3. Trains the model on GPU
4. Saves results back to Drive

## Setup Required (One Time)

1. **Upload VRD dataset to Drive**:
   - Go to Google Drive
   - Create folder: `vrd/`
   - Upload inside it:
     - `sg_train_images/` (folder with ~4,000 images)
     - `sg_test_images/` (folder with ~1,000 images)
     - `sg_train_annotations.json`
     - `sg_test_annotations.json`

2. **How to Use This Notebook**:
   - Go to: https://colab.research.google.com
   - File → New notebook
   - Copy the cells below
   - Runtime → Change runtime type → GPU (T4)
   - Runtime → Run all
   - Wait ~25-30 min

---

## CELL 1: Clone Repository

```python
# Clone repo from GitHub
import os

REPO_URL = "https://github.com/hajteyib/Visual-Spatial-Contrastive-Learning.git"
REPO_NAME = "Visual-Spatial-Contrastive-Learning"

# Remove if exists (for reruns)
if os.path.exists(REPO_NAME):
    !rm -rf {REPO_NAME}

!git clone {REPO_URL}
%cd {REPO_NAME}

print("\\n✅ Repository cloned!")
```

---

## CELL 2: Install Dependencies

```python
# Install requirements
!pip install -q -r requirements.txt

print("✅ Dependencies installed!")
```

---

## CELL 3: Verify GPU

```python
# Check GPU availability
import torch

if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Go to: Runtime > Change runtime type > Select GPU")
```

---

## CELL 4: Mount Drive & Link Dataset

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Link VRD dataset from Drive
DATASET_PATH = "/content/drive/MyDrive/vrd"

if os.path.exists(DATASET_PATH):
    # Create symbolic link
    !ln -s {DATASET_PATH} vrd
    
    print(f"✅ Dataset linked from: {DATASET_PATH}")
    print("\\nDataset contents:")
    !ls vrd/
    
    # Verify files exist
    required_files = [
        'vrd/sg_train_images',
        'vrd/sg_test_images', 
        'vrd/sg_train_annotations.json',
        'vrd/sg_test_annotations.json'
    ]
    
    all_good = True
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ Missing: {f}")
            all_good = False
    
    if all_good:
        print("\\n✅ All dataset files found!")
    else:
        print("\\n❌ Some files are missing. Check your Drive folder.")
else:
    print(f"❌ ERROR: Dataset not found at {DATASET_PATH}")
    print(f"   Please upload VRD dataset to Google Drive!")
    print(f"   Expected location: MyDrive/vrd/")
```

---

## CELL 5: RUN TRAINING

```python
# Run training
print("\\n" + "="*70)
print("STARTING TRAINING - EXP #5")
print("6 Merged Classes + Long Training (50 epochs)")
print("="*70 + "\\n")

!python src/train.py

print("\\n" + "="*70)
print("TRAINING COMPLETED")
print("="*70)
```

---

## CELL 6: Save Results to Drive

```python
# Save results to Google Drive
import shutil

# Create results directory
RESULTS_DIR = "/content/drive/MyDrive/Spatial_Relations_Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Find latest experiment
checkpoints = sorted([d for d in os.listdir('checkpoints') if d.startswith('exp_')])

if checkpoints:
    latest_exp = checkpoints[-1]
    source = f"checkpoints/{latest_exp}"
    destination = f"{RESULTS_DIR}/{latest_exp}"
    
    # Copy to Drive
    shutil.copytree(source, destination)
    
    print(f"✅ Results saved to Google Drive:")
    print(f"   {destination}")
    print(f"\\nFiles saved:")
    !ls -lh {destination}
    
    # Show training history
    history_file = f"{destination}/training_history.txt"
    if os.path.exists(history_file):
        print(f"\\n📊 Training History (last 5 epochs):")
        !tail -5 {history_file}
else:
    print("⚠️  No experiment checkpoints found")
```

---

## CELL 7: Download Results (Optional)

```python
# Uncomment to download results directly to your computer
# (Alternatively, just access them from Google Drive)

# from google.colab import files
# 
# if checkpoints:
#     latest_exp = checkpoints[-1]
#     
#     # Download best models
#     files.download(f"checkpoints/{latest_exp}/best_visual_encoder.pth")
#     files.download(f"checkpoints/{latest_exp}/best_spatial_encoder.pth")
#     
#     # Download training history
#     files.download(f"checkpoints/{latest_exp}/training_history.txt")
#     
#     # Download loss plots
#     files.download(f"checkpoints/{latest_exp}/loss_final.png")
```

---

## Expected Runtime

- Clone + Setup: ~1 min
- Training (50 epochs): ~20-25 min on T4 GPU
- Total: ~25-30 min

## After Training

Results will be in:
- **Google Drive**: `MyDrive/Spatial_Relations_Results/exp_YYYYMMDD_HHMMSS/`
- Files:
  - `best_visual_encoder.pth` (best model)
  - `best_spatial_encoder.pth` (best model)
  - `training_history.txt` (all epochs)
  - `loss_final.png` (loss curves)

To analyze results:
1. Download from Drive
2. On your Mac: `python src/evaluate.py checkpoints/exp_YYYYMMDD_HHMMSS`
