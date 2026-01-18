# Exp #5 Révisé - TOP-5 Classes

**Date** : 2026-01-16

## Objectif

Maximiser performance avec classes **prouvées fonctionnelles**.

**Target** : 65-70% accuracy

---

## Configuration

### Classes (TOP-5)
```python
target_relations = [
    'on',       # 79% F1 (Exp #2), 13,071 samples
    'above',    # 70% F1, 4,995 samples
    'under',    # 52% F1, 2,951 samples
    'next to',  # 42% F1, 4,108 samples
    'below'     # 16% F1, 1,680 samples
]

Total: 26,805 train samples
Ratio: 7.8:1 (excellent équilibre)
```

### Architecture
- **Backbone** : ResNet-18 (frozen, 11M params)
- **Projection** : 1024→512→256, dropout 0.4
- **Spatial** : 8→64→128→256, dropout 0.3
- **Loss** : Supervised Contrastive
- **Sampler** : Balanced Batch (5 classes)

### Training
- **Epochs** : 50
- **Batch** : 24
- **LR** : 1e-4
- **Patience** : 12
- **Min epochs** : 30
- **Device** : MPS (Mac GPU)

---

## Différences vs Exp #5 Échec

| Aspect | Exp #5 Échec | Exp #5 Révisé |
|--------|--------------|---------------|
| **Classes** | 6 fusionnées | 5 directes ✅ |
| **Géométrie** | Mélangée | Cohérente ✅ |
| **Val Acc E1** | 8.3% ❌ | >20% attendu ✅ |

**Clé** : Pas de fusion = géométrie cohérente = loss fonctionne

---

## Success Criteria

**Epoch 1** : Val Acc > 20% (vs 8% échec)  
**Epoch 10** : Val Acc > 40%  
**Final** : Accuracy 65-70%  

**Minimum** : Beat 62.40% (Exp #2)

---

## Commande

```bash
cd /Users/hajteyibebou/Documents/MSI-Projet_Spatial_Relations
source venv/bin/activate
python -m src.train
```

**Durée** : ~2-2.5h
