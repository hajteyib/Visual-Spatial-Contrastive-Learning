# Exp #6 - TOP-5 Classes + InfoNCE Loss

**Date** : 2026-01-16

## Objectif

Utiliser la **loss qui marche** (InfoNCE from Exp #1) avec les **meilleures classes** (TOP-5).

**Target** : 63-67% accuracy

---

## Configuration

### Classes (TOP-5)
```python
target_relations = [
    'on',       # 79% F1, 13,071 train samples
    'above',    # 70% F1,  4,995 samples
    'under',    # 52% F1,  2,951 samples
    'next to',  # 42% F1,  4,108 samples
    'below'     # 16% F1,  1,680 samples
]

Total: 26,805 train samples
Ratio: 7.8:1 (excellent)
```

### Architecture
- **Backbone** : ResNet-18 (frozen, 11M params)
- **Projection** : 1024→512→256, dropout 0.4
- **Spatial** : 8→64→128→256, dropout 0.3
- **Loss** : **InfoNCE (self-supervised)** ← CLEF !
- **Sampler** : Random shuffle

### Training
- **Epochs** : 50
- **Batch** : 24
- **LR** : 1e-4
- **Patience** : 12
- **Min epochs** : 30

---

## Différences vs Exp #1 (Baseline 61.67%)

| Aspect | Exp #1 | Exp #6 |
|--------|---------|--------|
| **Classes** | 10 | 5 (TOP only) ✅ |
| **Loss** | InfoNCE ✅ | InfoNCE ✅ |
| **Sampling** | Random ✅ | Random ✅ |
| **Epochs** | 30 | 50 ✅ |
| **Classes** | All | Best only ✅ |

**Amélioration** : Moins de classes difficiles → meilleure accuracy attendue

---

## Pourquoi InfoNCE Marche

**InfoNCE (Self-supervised)** :
```python
Positifs = (visual, spatial) du MÊME sample
Négatifs = Toutes les autres paires du batch

Ne dépend PAS des labels
→ Marche avec n'importe quel sampling
→ PROUVÉ dans Exp #1 : 61.67% ✅
```

**Supervised Contrastive (Échec)** :
```python
Positifs = Samples avec MÊME label dans batch
→ Avec Balanced Sampler : seulement ~4 samples/classe
→ Très peu de positifs
→ Loss ne peut pas apprendre ❌

Résultat : 11% accuracy (Exp #5)
```

---

## Success Criteria

**Epoch 1** : Val Acc > 20% (vs 8% supervised échec)  
**Epoch 10** : Val Acc > 40%  
**Final** : 63-67%

**Minimum** : Beat 62.40% (Exp #2)

---

## Commande

```bash
cd ~/Documents/MSI-Projet_Spatial_Relations
source venv/bin/activate
python -m src.train
```

**Durée** : ~2-2.5h  
**Test local** : ✅ InfoNCE loss = 1.46
