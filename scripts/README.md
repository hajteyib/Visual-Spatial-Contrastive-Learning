# Scripts Utilitaires

Ce dossier contient les scripts d'analyse et d'évaluation du projet.

## 📊 Scripts d'Analyse VRD

### `analyze_dataset.py`
Analyse statistique complète du dataset VRD.
```bash
python scripts/analyze_dataset.py
```
**Output** : Distribution des classes, statistiques géométriques

### `analyze_all_relations.py`
Liste toutes les relations spatiales dans VRD.
```bash
python scripts/analyze_all_relations.py
```

### `visualize_vrd.py`
Visualise des échantillons du dataset VRD.
```bash
python scripts/visualize_vrd.py
```

## 🔍 Évaluation Cross-Dataset (PSG)

### `evaluate_psg_full56.py`
Évalue le modèle VRD sur toutes les 56 classes PSG.
```bash
python scripts/evaluate_psg_full56.py
```

**Résultats** :
- Accuracy: 35.27%
- Recall@5: 72.87%
- Recall@10: 84.26%

### `download_psg_images.py`
Télécharge les images PSG nécessaires (subset COCO val2017).
```bash
python scripts/download_psg_images.py
```

## 📁 Organisation des Résultats

Les résultats d'évaluation sont sauvegard

és dans `experiments/results/`
