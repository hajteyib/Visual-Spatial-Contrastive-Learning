# EXPERIMENT #3 - EfficientNet-B0

## Date
2025-12-17

## Objective
Beat Exp #1 (ResNet-18: 61.67%) with faster, more efficient architecture.  
**Target: 63-65% accuracy in 2-3h** (vs 6-8h ResNet-50)

---

## Architecture Change

### Visual Encoder

**Before (Exp #1/#2):**
- ResNet-18: 11M params, 512D features → 61.67%
- ResNet-50: 23M params, 2048D features → 62.40% (too slow)

**Now (Exp #3):**
- **EfficientNet-B0**: 5.3M params, 1280D features
- Projection: 2560D → 1024D → **256D** (doubled embedding size)
- **Frozen backbone** (transfer learning)

**Why EfficientNet-B0:**
1. ✅ **SOTA efficiency**: Best accuracy/params ratio
2. ✅ **Faster**: 2-3h training (vs 6-8h ResNet-50)
3. ✅ **Lightweight**: 5.3M params (vs 23M ResNet-50)
4. ✅ **Better features**: Modern architecture (2019 vs 2015)

---

## Configuration

```python
# config.py
BATCH_SIZE = 24           # Optimal for EfficientNet
EMBEDDING_DIM = 256       # Doubled (128→256)
EPOCHS = 25               # Quick test
LEARNING_RATE = 1e-4      # Standard
```

### Parameters

|  | Trainable | Frozen | Total |
|--|-----------|--------|-------|
| Visual Encoder | ~1.3M | 5.3M | 6.6M |
| Spatial Encoder | ~34k | 0 | 34k |
| **Total** | **~1.3M** | **~5.3M** | **~6.7M** |

**vs Exp #2 (ResNet-50)**: 4.4M trainable, 23.5M frozen (28M total)  
→ **74% fewer parameters** !

---

## Training Strategy

### Unchanged (Proven)
- Loss: InfoNCE contrastive
- Optimizer: Adam 1e-4
- Split: 70/20/10
- Augmentation: Flip + Color Jitter (train only)
- Early stopping: patience 3, min 10 epochs

### Key Difference
- **256D embeddings** (vs 128D) → More capacity
- **Faster epochs** (~2 min vs ~4 min ResNet-50)

---

## Expected Results

### Conservative
- Accuracy: **63-64%** (+1-2% vs Exp #1)
- Time: **2.5-3h**
- Per-class: More balanced than Exp #1

### Optimistic  
- Accuracy: **65-67%** (+3-5% vs Exp #1)
- Time: **2-2.5h**
- F1 Weighted: 62-65%

---

## Success Criteria

✅ **Minimum**: 63% accuracy (beats ResNet-18)  
✅ **Target**: 65% accuracy  
✅ **Speed**: <3h training time  
✅ **Healthy**: Train/Val gap <0.6

---

## Comparison Table

| Metric | Exp #1 (R18) | Exp #2 (R50) | **Exp #3 (EffNet)** |
|--------|--------------|--------------|---------------------|
| Accuracy | 61.67% | 62.40% | **Target: 65%** |
| Params | 15M | 28M | **6.7M** (-76%) |
| Training | 3-4h | 6-8h | **2-3h** (-50%) |
| Embedding | 128D | 128D | **256D** (+100%) |
| Epoch time | ~3min | ~4min | **~2min** (-50%) |

---

## Launch Command

```bash
cd /Users/hajteyibebou/Documents/MSI-Projet_Spatial_Relations
source venv/bin/activate
python3 train.py
```

**Checkpoint**: `checkpoints/exp_YYYYMMDD_HHMMSS/`

---

## Post-Training

1. Run `evaluate.py` on best checkpoint
2. Compare with Exp #1 (61.67%) and Exp #2 (62.40%)
3. If >63%: **SUCCESS**, EfficientNet validated
4. If <63%: Fallback to ResNet-18 optimized
