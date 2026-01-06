# EXPERIMENT #4 - ResNet-18 + Supervised Contrastive + Balanced Sampling

## Date
2025-12-18

## ✅ VERIFICATION COMPLETE

### What Changed from Exp #3
| Aspect | Exp #3 (EfficientNet) | **Exp #4** |
|--------|----------------------|------------|
| Backbone | EfficientNet-B0 (5.3M) | **ResNet-18 (11M)** ✅ |
| Contrastive Loss | Self-supervised | **Supervised** ✅ |
| Batch Sampling | Random | **Balanced** ✅ |
| Dropout | 0.3 | **0.4/0.3** ✅ |
| Epochs | 25 | **30** ✅ |
| Result | 56.54% | **Target: 63-68%** |

---

## Architecture

**Visual Encoder** :
- ResNet-18 (ImageNet, FROZEN)
- Features: 512D (sujet) + 512D (objet) = 1024D
- Projection: 1024D → 512D → 256D
- Dropout: **0.4** (augmenté)

**Spatial Encoder** :
- MLP: 8D → 64D → 128D → 256D
- Dropout: **0.3** (augmenté)

**Params** :
- Trainable: 698k (6%)
- Frozen: 11.2M (94%)

---

## Innovations Exp #4

### 1. Supervised Contrastive Loss ⭐

**Avant (Self-supervised)** :
```python
# Positifs : Même sample apparié
# Négatifs : Autres samples aléatoires
→ N'utilise PAS les labels
```

**Maintenant (Supervised)** :
```python
# Positifs : TOUS les samples avec MÊME label
# Négatifs : TOUS les samples avec labels DIFFÉRENTS
→ Utilise les labels pour meilleurs positifs/négatifs
```

**Gain attendu** : +3-5%

---

### 2. Balanced Batch Sampler ⚖️

**Avant** :
```
Batch random : [on, on, on, on, above, on, on, ...]
→ 70% "on", classes rares noyées
```

**Maintenant** :
```
Batch balancé : [on, on, above, above, under, under, ...]
→ ~10% chaque classe, équilibré
```

**Résultat** :
- 341 batches/epoch (vs 1416 avant)
- ~2 samples/classe/batch
- Toutes classes vues équitablement

**Gain attendu** : +2-4% sur classes rares

---

## Configuration

```python
# config.py
BATCH_SIZE = 24
EMBEDDING_DIM = 256
EPOCHS = 30
LEARNING_RATE = 1e-4

# Dataset
Classes: 10 (in/over, PAS inside/outside)
Train: 33,978 samples
Val: 9,708 samples
Test: 4,855 samples

# Regularization
- ResNet-18 frozen (94%)
- Dropout 0.4 (projection)
- Dropout 0.3 (spatial)
- Early stopping: patience 7, min 20 epochs
```

---

## Launch Command

```bash
cd /Users/hajteyibebou/Documents/MSI-Projet_Spatial_Relations
source venv/bin/activate
python3 train.py
```

**IMPORTANT** : Vérifiez la sortie :
```
✅ Train: 33978 samples  ← Doit être 33978 !
--- Configuration Balanced Batch Sampler ---
  Batches par epoch: 341  ← Moins que avant (normal)
  Samples par classe/batch: ~2
```

---

## Expected Output

```
Epoch 1/30: 100%|████| 341/341 [02:30<00:00]
   -> Train Loss: 2.XXXX | Val Loss: 2.XXXX | Val Acc: 0.XXX
      🔥 Meilleur modèle sauvegardé !

Epoch 2/30: ...
```

**Durée** :
- ~2.5 min/epoch (batchs balancés = moins de batches)
- Total: ~75 min (1h15) si 30 epochs
- Probable early stopping ~epoch 22-25 → ~55-65 min

---

## Success Criteria

**Minimum** : 63% accuracy (battre Exp #1's 61.67%)  
**Target** : 65-66% accuracy  
**Optimal** : 68%+ accuracy  

**Classes rares** :
- over, left of, right of, near : >20% recall (vs 0-4% avant)
- All classes : >10% recall

---

## After Training

```bash
# Evaluer avec SVM
python3 evaluate.py checkpoints/exp_YYYYMMDD_HHMMSS
```

Résultats dans : `checkpoints/exp_YYYYMMDD_HHMMSS/evaluation_results/`

---

## Key Differences vs Previous Exps

**vs Exp #1 (ResNet-18, 61.67%)** :
- ✅ Supervised contrastive (meilleur)
- ✅ Balanced sampling (équilibre)
- ✅ Dropout plus fort (régularisation)

**vs Exp #3 (EfficientNet-B0, 56.54%)** :
- ✅ ResNet-18 (prouvé meilleur pour notre tâche)
- ✅ Supervised contrastive (vs self-supervised)
- ✅ Balanced sampling (équilibre)

---

**Tout est vérifié et testé - Prêt à lancer !** 🚀
