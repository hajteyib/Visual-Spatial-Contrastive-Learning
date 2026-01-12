# Colab Training Notebook
# Save this as: train_gpu.ipynb

Upload this to Google Colab and it will:
1. Clone your GitHub repo
2. Install dependencies  
3. Link VRD dataset from your Drive
4. Train on GPU
5. Save results back to Drive

## Instructions:

1. Go to: https://colab.research.google.com
2. File > Upload notebook
3. Upload the train_gpu.ipynb (create from content below)
4. Runtime > Change runtime type > GPU (T4)
5. Runtime > Run all

## Notebook Content (copy to .ipynb):

```json
{
 "cells": [
  {"cell_type": "code", "source": ["!git clone https://github.com/hajteyib/Visual-Spatial-Contrastive-Learning.git\\n%cd Visual-Spatial-Contrastive-Learning\\n!pip install -q -r requirements.txt"]},
  {"cell_type": "code", "source": ["import torch\\nprint(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"]},
  {"cell_type": "code", "source": ["from google.colab import drive\\ndrive.mount('/content/drive')\\n!ln -s /content/drive/MyDrive/VRD_Dataset vrd"]},
  {"cell_type": "code", "source": ["!python src/train.py"]},
  {"cell_type": "code", "source": ["import shutil, os\\nos.makedirs('/content/drive/MyDrive/Results', exist_ok=True)\\nlatest = sorted([d for d in os.listdir('checkpoints') if d.startswith('exp_')])[-1]\\nshutil.copytree(f'checkpoints/{latest}', f'/content/drive/MyDrive/Results/{latest}')"]}
 ]
}
```

## Setup on Google Drive:
- Upload VRD dataset to: MyDrive/VRD_Dataset/
- Results will be in: MyDrive/Results/
