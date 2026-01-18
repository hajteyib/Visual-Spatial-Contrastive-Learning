# Experiments Documentation

Ce dossier contient la documentation de toutes les expériences menées lors du projet.

## 📊 Vue d'Ensemble des Expériences

| Exp | Objectif | Architecture | Loss | Classes | Résultat |
|-----|----------|--------------|------|---------|----------|
| #1 | Baseline | ResNet-18 | InfoNCE | 10 VRD | **61.67%** |
| #2 | Amélioration | ResNet-50 | SupCon | 10 VRD | **62.40%** ⭐ |
| #3 | Architecture efficace | EfficientNet-B0 | InfoNCE | 10 VRD | 56.54% |

## 🎯 Meilleur Modèle

**Exp #2 - ResNet-50 + Supervised Contrastive**
- Test Accuracy: **62.40%**
- F1-Score moyen: 0.57
- Checkpoint: `exp_20251202_175017`

## �� Structure

```
experiments/
├── README.md                    ← Ce fichier
├── EXP2_CONFIG.md              ← Documentation Exp #2
├── EXP3_EFFICIENTNET_CONFIG.md ← Documentation Exp #3
└── results/
    ├── evaluation_summary.txt   ← Évaluation PSG 56 classes
    └── psg_spatial14_results.txt ← Évaluation PSG 14 spatiales
```

## 📦 Checkpoints (non inclus dans GitHub)

Les checkpoints sont trop volumineux pour GitHub (> 100 MB chacun).

**Structure des checkpoints** :
```
checkpoints/exp_YYYYMMDD_HHMMSS/
├── config.txt                    ← Configuration expérience
├── training_history.txt          ← Loss/Accuracy par epoch
├── best_visual_encoder.pth       ← Modèle visuel (ResNet)
├── best_spatial_encoder.pth      ← Modèle spatial (MLP)
└── evaluation_results/
    ├── results.txt               ← Résultats test
    └── confusion_matrix.png      ← Matrice confusion
```

**Disponibilité** : Sur demande
