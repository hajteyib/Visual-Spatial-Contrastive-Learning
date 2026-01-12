# Experiment #5 - 6 Merged Classes + Long Training

## Date: 2026-01-12

## Objective
Maximize accuracy by merging similar classes and allowing long training convergence.

**Target**: 70-75% accuracy

---

## Changes

### Dataset - 6 Merged Classes
```
on              : 1,839 samples
vertical_above  :   869 samples (above + over)
vertical_below  :   684 samples (below + under)
proximity       :   934 samples (near + next to)
horizontal      :   220 samples (left of + right of)
in              :   309 samples

Imbalance: 8:1 (vs 19:1 before) ✅
```

### Training
- EPOCHS: 50 (was 30)
- PATIENCE: 12 (was 7)
- MIN_EPOCHS: 30 (was 20)

### Fixes
- Loss plot Y-axis starts at 0

---

## Files Modified
- `src/dataset.py`: Class merging
- `src/config.py`: 50 epochs
- `src/train.py`: Plot fix + early stopping

---

## Training
Local: `python src/train.py`  
Colab: See `COLAB_TRAINING.md`

**Runtime**: ~25 min on T4 GPU

---

## Expected Results
Conservative: 68-72%  
Optimistic: 73-76%

Minimum success: >65%
