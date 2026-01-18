# Expérience #2 - Configuration et Objectifs

## 🎯 Objectifs de l'Expérience

**Améliorer** l'Expérience #1 (Baseline 61.67%) en :
1. Combattant le déséquilibre extrême des classes (344:1 ratio)
2. Augmentant la capacité du modèle
3. Optimisant la convergence

**Cible** : 73-79% accuracy (+12-18%)

---

## 🔧 Modifications par Rapport à Exp #1

### 1. Architecture Modèle

**Visual Encoder - ResNet-50** (au lieu de ResNet-18)
- Backbone gelé : 25M paramètres
- Features : 2048D (au lieu de 512D) → 4x plus riche
- Projection : 4096 → 1024 → 128

**Paramètres Totaux** :
- ResNet-50 gelé : ~25M (non-entraînables)
- Projection Visual : ~4.7M (entraînables)
- Spatial Encoder : 73k (entraînables)
- **Total entraînable : ~4.77M** (vs 615k dans Exp #1)

### 2. Loss Function

**Weighted CrossEntropyLoss**

Poids par classe (basés sur analyse dataset) :
```python
on:       0.24   # Classe dominante (13103 exemples)
under:    1.06
above:    0.62
below:    1.77
left of:  3.92
right of: 4.72
near:     1.28
next to:  0.75
inside:   22.53  # Classe rare (137 exemples)
outside:  81.21  # Classe très rare (38 exemples)
```

**Impact** : Pénalise 81x plus les erreurs sur "outside" vs "on"

### 3. Data Balancing

**Oversampling Ciblé (train uniquement)** :
- Classes <100 exemples → dupliquer x3
- Classes 100-500 exemples → dupliquer x2
- Classes >500 exemples → aucune duplication

**Résultat attendu** :
- Train : ~40,000 paires (vs 30,860 dans Exp #1)
- inside : 137 → 411 exemples
- outside : 38 → 114 exemples

### 4. Training Optimizations

**LR Scheduler** : ReduceLROnPlateau
- Factor : 0.5 (divise LR par 2)
- Patience : 3 epochs
- Min LR : 1e-6

**Gradient Clipping** : max_norm=1.0
- Évite explosions avec weighted loss

**Early Stopping Amélioré** :
- Patience : 5 epochs (vs 3 dans Exp #1)
- Min epochs : 15 (vs 10 dans Exp #1)

**Epochs Max** : 50 (vs 15 dans Exp #1)

---

## 📊 Hyperparamètres

```python
BATCH_SIZE = 16           # Identique
LEARNING_RATE = 1e-4      # Identique (initial)
EPOCHS = 50               # Augmenté
IMG_SIZE = 128            # Identique
EMBEDDING_DIM = 128       # Identique
TEMPERATURE = 0.07        # Identique
DROPOUT = 0.3 / 0.2       # Identique
```

---

## 📈 Prédictions Performance

### Accuracy Globale

- **Conservative** : 73-76% (+12-15%)
- **Optimiste** : 76-79% (+15-18%)

### Par Relation (Prédictions)

| Relation | Exp #1 | Exp #2 (Prédit) | Gain |
|----------|--------|-----------------|------|
| on | 91% | 88-90% | -1 à -3% (acceptable) |
| above | 68% | 72-75% | +4-7% |
| under | 59% | 65-70% | +6-11% |
| next to | 44% | 50-55% | +6-11% |
| **below** | **10%** | **30-40%** | **+20-30%** ⭐ |
| **near** | **3%** | **15-25%** | **+12-22%** ⭐ |
| **left of** | **2%** | **20-30%** | **+18-28%** ⭐ |
| **right of** | **0%** | **15-25%** | **+15-25%** ⭐ |
| **inside** | **0%** | **10-20%** | **+10-20%** ⭐ |
| **outside** | **0%** | **5-15%** | **+5-15%** ⭐ |

### Convergence Attendue

- Meilleur modèle : Epoch 18-25 (vs 12 dans Exp #1)
- Val Loss finale : 1.1-1.2 (vs 1.45 dans Exp #1)
- Val Acc finale : 52-58% (vs 48% dans Exp #1)

---

## ⏱️ Temps Estimé

**Par epoch** : 12-15 minutes (vs 8-10 min dans Exp #1)
- ResNet-50 est plus lourd
- Oversampling augmente dataset

**Total** : 5-7 heures
- Si early stopping à epoch 25 : ~5h
- Si 50 epochs complets : ~10h

---

## 🔬 Modifications Code

### model.py
- Ligne 12 : `resnet18` → `resnet50`
- Ligne 21-24 : Projection adaptée (4096 → 1024 → 128)

### config.py
- Ligne 11 : `EPOCHS = 15` → `EPOCHS = 50`

### dataset.py
- Lignes 97-130 : Oversampling ciblé (nouveau)

### train.py
- Lignes 123-150 : Weighted loss + LR scheduler
- Ligne 162-165 : Gradient clipping
- Lignes 195-205 : Early stopping amélioré

---

## ✅ Checklist Pré-Entraînement

- [x] ResNet-50 implémenté
- [x] Weighted loss configurée
- [x] Oversampling activé
- [x] LR scheduler ajouté
- [x] Gradient clipping activé
- [x] Early stopping amélioré
- [x] Epochs augmentés à 50
- [ ] Lancer `python3 train.py`

---

## 📝 Notes pour Analyse Post-Entraînement

**Comparer avec Exp #1** :
- Accuracy globale
- Recall par relation (focus sur rares)
- Val Loss finale
- Nombre d'epochs avant convergence
- Impact du LR scheduler (observer les changements de LR)

**Vérifier** :
- Pas de sur-apprentissage (écart train/val)
- Amélioration sur classes rares (below, left of, right of, etc.)
- Stabilité de la convergence

---

**Date de création** : 2 Décembre 2025  
**Prêt à lancer** ✅
