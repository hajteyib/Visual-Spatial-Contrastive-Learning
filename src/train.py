import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Import des fichiers locaux
import config
from dataset import VRDDataset
from model import VisualEncoder, SpatialEncoder

# === BALANCED BATCH SAMPLER (Exp #4) ===
class BalancedBatchSampler(Sampler):
    """
    Échantillonneur qui balance les classes dans chaque batch.
    Garantit que chaque batch contient ~égal nombre de chaque classe.
    """
    def __init__(self, dataset, batch_size=24, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_classes = len(dataset.rel2idx)
        
        # Grouper les indices par classe
        self.class_indices = {i: [] for i in range(self.num_classes)}
        for idx, sample in enumerate(dataset.samples):
            label = dataset.rel2idx[sample['predicat']]
            self.class_indices[label].append(idx)
        
        # Calculer nombre de samples par classe dans un batch
        self.samples_per_class = max(1, batch_size // self.num_classes)
        
        # Calculer nombre total de batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
    def __iter__(self):
        # Mélanger les indices de chaque classe
        shuffled_indices = {
            cls: indices.copy() for cls, indices in self.class_indices.items()
        }
        for indices in shuffled_indices.values():
            random.shuffle(indices)
        
        # Créer les batches
        for batch_idx in range(self.num_batches):
            batch = []
            for cls in range(self.num_classes):
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(shuffled_indices[cls][start:end])
            
            random.shuffle(batch)  # Mélanger le batch
            yield batch
    
    def __len__(self):
        return self.num_batches


# === SUPERVISED CONTRASTIVE LOSS (Exp #4) ===
def supervised_contrastive_loss(z_v, z_s, labels, temperature=0.07):
    """
    Loss contrastive supervisée qui utilise les labels.
    
    Positifs : Paires avec MÊME label (même relation spatiale)
    Négatifs : Paires avec labels DIFFÉRENTS
    
    Args:
        z_v: Embeddings visuels normalisés [batch, 256]
        z_s: Embeddings spatiaux normalisés [batch, 256]
        labels: Labels [batch] (0-9 pour 10 classes)
        temperature: Température pour scaling
    """
    batch_size = z_v.size(0)
    device = z_v.device
    
    # Similarités entre visual et spatial
    sim_matrix = torch.matmul(z_v, z_s.T) / temperature  # [batch, batch]
    
    # Masque des paires positives (même label)
    labels = labels.unsqueeze(1)  # [batch, 1]
    pos_mask = (labels == labels.T).float()  # [batch, batch]
    pos_mask.fill_diagonal_(0)  # Exclure self-similarity
    
    # Nombre de positifs par sample
    num_positives = pos_mask.sum(dim=1)  # [batch]
    
    # Exponentielle des similarités
    exp_sim = torch.exp(sim_matrix)
    
    # Somme des similarités avec positifs
    pos_sum = (exp_sim * pos_mask).sum(dim=1)  # [batch]
    
    # Somme totale (exclure self)
    total_sum = exp_sim.sum(dim=1) - torch.exp(torch.diagonal(sim_matrix))  # [batch]
    
    # Loss : -log(pos_sum / total_sum)
    loss = -torch.log((pos_sum + 1e-8) / (total_sum + 1e-8))
    
    # Moyenne seulement sur samples ayant des positifs
    valid_mask = num_positives > 0
    if valid_mask.sum() > 0:
        loss = loss[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss


def validate(visual_model, spatial_model, dataloader, device):
    """ Calcule la loss et accuracy sur validation. """
    visual_model.eval()
    spatial_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for img_s, img_o, spatial_vec, labels in dataloader:
            img_s, img_o = img_s.to(device), img_o.to(device)
            spatial_vec, labels = spatial_vec.to(device), labels.to(device)
            
            # Forward
            z_v = visual_model(img_s, img_o)
            z_s = spatial_model(spatial_vec)
            
            # Loss
            loss = supervised_contrastive_loss(z_v, z_s, labels)
            total_loss += loss.item()
            
            # Accuracy (similarité maximale)
            sim_matrix = torch.matmul(z_v, z_s.T)
            predictions = sim_matrix.argmax(dim=1)
            correct_predictions = torch.arange(labels.size(0), device=device)
            total_correct += (predictions == correct_predictions).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, save_path):
    """ Sauvegarde la courbe de loss avec axe Y commençant à 0. """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    # FIX Exp #5: Forcer Y-axis à démarrer à 0
    plt.ylim(bottom=0)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def train():
    print("=" * 70)
    print("ENTRAÎNEMENT CONTRASTIF - RELATIONS SPATIALES (Exp #5)")
    print("=" * 70)
    
    # --- 1. Dossier d'expérience ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.CHECKPOINT_DIR, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n📁 Dossier d'expérience : {exp_dir}\n")
    
    # --- 2. Chargement des données ---
    print("--- Chargement des données (Split 70/20/10) ---")
    train_ds = VRDDataset(subset='train')
    val_ds = VRDDataset(subset='val')
    test_ds = VRDDataset(subset='test')
    
    print(f"✅ Train: {len(train_ds)} samples")
    print(f"✅ Val: {len(val_ds)} samples")
    print(f"✅ Test: {len(test_ds)} samples (réservé pour évaluation finale)")
    
    # --- 3. Dataloaders avec Balanced Sampler ---
    print("\n--- Configuration Balanced Batch Sampler ---")
    train_sampler = BalancedBatchSampler(train_ds, batch_size=config.BATCH_SIZE)
    print(f"  Batches par epoch: {len(train_sampler)}")
    print(f"  Samples par classe/batch: ~{train_sampler.samples_per_class}")
    
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=0
    )
    
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Sauvegarde config
    config_path = os.path.join(exp_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"=== EXP #5 - 6 MERGED CLASSES + LONG TRAINING ===\n\n")
        f.write(f"Dataset:\n")
        f.write(f"  - Train: {len(train_ds)} samples\n")
        f.write(f"  - Val: {len(val_ds)} samples\n")
        f.write(f"  - Test: {len(test_ds)} samples\n\n")
        f.write(f"Hyperparamètres:\n")
        f.write(f"  - Batch size: {config.BATCH_SIZE}\n")
        f.write(f"  - Learning rate: {config.LEARNING_RATE}\n")
        f.write(f"  - Epochs max: {config.EPOCHS}\n")
        f.write(f"  - Embedding dim: {config.EMBEDDING_DIM}\n\n")
        f.write(f"Architecture:\n")
        f.write(f"  - Visual: ResNet-18 frozen + Projection (dropout 0.4)\n")
        f.write(f"  - Spatial: MLP 8→64→128→256 (dropout 0.3)\n")
        f.write(f"  - Loss: Supervised Contrastive\n")
        f.write(f"  - Sampler: Balanced Batch Sampler\n")
    print(f"📝 Configuration sauvegardée : {config_path}")
    
    # --- 4. Modèles ---
    print(f"\n--- Initialisation des modèles sur {config.device} ---")
    visual_model = VisualEncoder(embedding_dim=config.EMBEDDING_DIM).to(config.device)
    spatial_model = SpatialEncoder(input_dim=8, embedding_dim=config.EMBEDDING_DIM).to(config.device)
    
    trainable_params = sum(p.numel() for p in visual_model.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in spatial_model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in visual_model.parameters() if not p.requires_grad)
    print(f"🔓 Paramètres entraînables : {trainable_params:,}")
    print(f"🔒 Paramètres gelés (ResNet-18): {frozen_params:,}")
    
    optimizer = optim.Adam([
        {'params': visual_model.parameters()},
        {'params': spatial_model.parameters()}
    ], lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 12  # Exp #5: Long training (was 7)
    min_epochs = 30  # Exp #5: Long training (was 20)
    
    history = {'train': [], 'val': [], 'val_acc': []}
    
    # --- 5. Entraînement ---
    print(f"\n--- Démarrage Entraînement ({config.EPOCHS} epochs max) ---\n")
    
    try:
        for epoch in range(config.EPOCHS):
            # TRAIN
            visual_model.train()
            spatial_model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
            
            for img_s, img_o, spatial_vec, labels in pbar:
                img_s, img_o = img_s.to(config.device), img_o.to(config.device)
                spatial_vec, labels = spatial_vec.to(config.device), labels.to(config.device)
                
                # Forward
                z_v = visual_model(img_s, img_o)
                z_s = spatial_model(spatial_vec)
                
                # Supervised Contrastive Loss
                loss = supervised_contrastive_loss(z_v, z_s, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
            epoch_train_loss = running_loss / len(train_loader)
            
            # VALIDATION
            epoch_val_loss, epoch_val_acc = validate(visual_model, spatial_model, val_loader, config.device)
            
            history['train'].append(epoch_train_loss)
            history['val'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            print(f"   -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.3f}")
            
            # Sauvegarde courbe
            plot_path = os.path.join(exp_dir, f"loss_epoch_{epoch+1:02d}.png")
            plot_curves(history['train'], history['val'], plot_path)
            
            # Sauvegarde meilleur modèle
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(visual_model.state_dict(), os.path.join(exp_dir, "best_visual_encoder.pth"))
                torch.save(spatial_model.state_dict(), os.path.join(exp_dir, "best_spatial_encoder.pth"))
                print("      🔥 Meilleur modèle sauvegardé !")
            else:
                patience_counter += 1
                print(f"      ⚠️  Pas d'amélioration ({patience_counter}/{patience})")
            
            # Early Stopping
            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n🛑 Early Stopping après {patience} epochs sans amélioration")
                print(f"   Meilleur Val Loss : {best_val_loss:.4f}")
                break
        
        print("\n✅ Entraînement terminé normalement !")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu manuellement (Ctrl+C)")
        print("💾 Sauvegarde des résultats partiels...")
    
    # Sauvegarde finale
    plot_path_final = os.path.join(exp_dir, "loss_final.png")
    plot_curves(history['train'], history['val'], plot_path_final)
    print(f"📈 Courbe finale sauvegardée : {plot_path_final}")
    
    # Historique
    history_path = os.path.join(exp_dir, "training_history.txt")
    with open(history_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc\n")
        for i in range(len(history['train'])):
            f.write(f"{i+1},{history['train'][i]:.4f},{history['val'][i]:.4f},{history['val_acc'][i]:.4f}\n")
    print(f"📊 Historique sauvegardé : {history_path}")
    
    print(f"\n🎯 Résultats finaux :")
    print(f"   - Meilleure Val Loss : {best_val_loss:.4f}")
    print(f"   - Tous les fichiers dans : {exp_dir}")


if __name__ == "__main__":
    train()