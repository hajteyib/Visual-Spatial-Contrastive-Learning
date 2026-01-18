# EXPERIMENT #3 - EfficientNet-B0 (CORRECTED)

## Date
2025-12-17 21:15

## ✅ PRE-LAUNCH VERIFICATION COMPLETE

### Configuration Verified

**dataset.py** ✅
```python
Classes: ['on', 'under', 'above', 'below', 'left of', 
          'right of', 'near', 'next to', 'in', 'over']
          
Total: 33,978 train samples
✅ 'in' présent (3,186 samples)
✅ 'over' présent (1,504 samples)
❌ 'inside' ABSENT (trop rare)
❌ 'outside' ABSENT (trop rare)
```

**model.py** ✅
```python
VisualEncoder: EfficientNet-B0 → 1280D → 256D
SpatialEncoder: 8D → 64D → 128D → 256D
Backbone: FROZEN (4M params)
Trainable: 2.9M params
```

**config.py** ✅
```python
BATCH_SIZE = 24
EMBEDDING_DIM = 256
EPOCHS = 25
LEARNING_RATE = 1e-4
```

---

## 🚀 LAUNCH COMMAND

```bash
cd /Users/hajteyibebou/Documents/MSI-Projet_Spatial_Relations
source venv/bin/activate
python3 train.py
```

---

## Expected Output

```
✅ Train: 33978 samples  ← DOIT être 33978 PAS 30860 !
✅ Val: 9708 samples
✅ Test: 4855 samples

🔓 Paramètres entraînables : 2,926,784
🔒 Paramètres gelés : 4,007,548

Epoch 1/25: [progress bar]
```

---

## Checkpoint Location

```
checkpoints/exp_YYYYMMDD_HHMMSS/
```

---

## Duration

- **~2-3h** total (EfficientNet rapide)
- **~2 min/epoch** (vs 4 min ResNet-50)

---

## Success Criteria

**MINIMUM** : >61.67% (beat ResNet-18 baseline)  
**TARGET** : 63-65%  
**OPTIMAL** : 66%+

---

## After Training

```bash
python3 evaluate.py checkpoints/exp_YYYYMMDD_HHMMSS
```

Will generate full metrics (accuracy, precision, recall, F1).

---

## Differences vs Previous Failed Run

| Aspect | Failed Run | **THIS RUN** |
|--------|-----------|--------------|
| Classes | inside/outside | **in/over** ✅ |
| Samples | 30,860 | **33,978** (+10%) ✅ |
| Imbalance | 260:1 | **~19:1** ✅ |

**This time it should work better!** 🎯
