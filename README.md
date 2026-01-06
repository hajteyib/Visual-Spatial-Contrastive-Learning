# Contrastive Spatial Relations

**Contrastive learning for spatial relations detection in Visual Relationship Detection (VRD)**

## 📋 Overview

This project explores contrastive learning approaches for detecting spatial relationships between objects in images. Using the Visual Relationship Detection (VRD) dataset, we train dual-encoder models that learn to align visual and geometric representations of spatial relations.

## 🎯 Spatial Relations

We focus on 10 spatial relations:
- **Vertical**: `above`, `below`, `on`, `under`, `over`
- **Horizontal**: `left of`, `right of`
- **Proximity**: `near`, `next to`
- **Containment**: `in`

## 🏗️ Architecture

**Dual-Encoder Contrastive Learning:**

```
Visual Encoder (ResNet-18)
  ├─ Crop Subject → Features (512D)
  ├─ Crop Object → Features (512D)
  └─ Projection → Embedding (256D)

Spatial Encoder (MLP)
  └─ Geometric Vector (8D) → Embedding (256D)

Loss: Supervised Contrastive
```

**Geometric Vector (8D):**
- Normalized displacement (dx, dy)
- Distance²
- Angle encoding (sin θ, cos θ)
- Area features (subject, object, ratio)

## 📊 Results

| Experiment | Architecture | Sampling | Accuracy |
|------------|-------------|----------|----------|
| Exp #1 | ResNet-18 + InfoNCE | Random | 61.67% |
| Exp #2 | ResNet-50 + InfoNCE | Oversampling | 62.40% |
| Exp #3 | EfficientNet-B0 | Random | 56.54% |
| **Exp #4** | **ResNet-18 + Supervised** | **Balanced** | **~65%*** |

*In progress

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision numpy matplotlib scikit-learn tqdm Pillow
```

### Dataset

Download the [VRD dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and place it in `vrd/`:

```
vrd/
├── images/
├── annotations_train.json
└── annotations_test.json
```

### Training

```bash
python train.py
```

Results will be saved in `checkpoints/exp_YYYYMMDD_HHMMSS/`

### Evaluation

```bash
python evaluate.py checkpoints/exp_YYYYMMDD_HHMMSS
```

## 📁 Project Structure

```
.
├── config.py              # Configuration and paths
├── dataset.py             # VRD dataset loader
├── model.py               # Visual & Spatial encoders
├── train.py               # Training script (contrastive)
├── evaluate.py            # Evaluation with SVM classifier
├── utils_geometry.py      # Geometric vector computation
└── checkpoints/           # Experiment results
```

## 🔬 Key Features

- **Supervised Contrastive Loss**: Uses labels to define positive/negative pairs
- **Balanced Batch Sampling**: Ensures equal representation of all classes
- **Frozen Backbone**: Transfer learning from ImageNet
- **8D Geometric Encoding**: Position, distance, angle, and size features

## 📈 Experiments

### Exp #1 - Baseline
- ResNet-18, InfoNCE loss, random sampling
- **Result**: 61.67% accuracy

### Exp #2 - ResNet-50
- Larger backbone, oversampling rare classes
- **Result**: 62.40% accuracy

### Exp #3 - EfficientNet-B0
- Efficient architecture, faster training
- **Result**: 56.54% accuracy (worse, features too compact)

### Exp #4 - Supervised + Balanced
- Supervised contrastive, balanced batch sampler
- **Status**: In progress

## 🎓 Academic Context

This project is part of a Master's program exploring contrastive learning for visual-spatial reasoning. The focus is on understanding how geometric and visual information can be jointly learned through contrastive objectives.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- VRD Dataset: [Visual Relationship Detection with Language Priors](https://cs.stanford.edu/people/ranjaykrishna/vrd/)
- PyTorch team
- ResNet, EfficientNet pre-trained models (ImageNet)

## 📧 Contact

[Your contact information]

---

**Note**: This is an exploratory research project. Results and implementations are subject to ongoing experimentation.
