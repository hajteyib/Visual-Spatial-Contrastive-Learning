import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Copie ici les valeurs de ton résultat (ou relance l'évaluation si tu veux être précis)
# Mais pour un rapport, tu peux aussi juste expliquer le texte.

# Voici le code si tu veux le lancer à la suite dans evaluate.py (ajoute le à la fin)
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    # Normalisation pour voir des pourcentages
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de Confusion (Normalisée)')
    plt.show()