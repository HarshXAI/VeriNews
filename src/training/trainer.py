"""
Training utilities for GAT models
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class GATTrainer:
    """Trainer for Graph Attention Networks"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        device: Optional[str] = None,
        checkpoint_dir: str = "./experiments"
    ):
        """
        Initialize trainer
        
        Args:
            model: GAT model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use (cuda, mps, or cpu)
            checkpoint_dir: Directory to save checkpoints
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        
        # Optimizer and scheduler
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = nn.NLLLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> dict:
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss
            loss = self.criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """
        Validate the model
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = batch.to(self.device)
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            # Track metrics
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Compute F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ):
        """
        Train the model
        
        Args:
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['f1'])
            
            # Print metrics
            print(f"\nEpoch {self.current_epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > self.best_val_metric:
                self.best_val_metric = val_metrics['f1']
                self.patience_counter = 0
                
                if save_best_only:
                    self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {self.current_epoch}")
                break
        
        print(f"\nTraining complete! Best F1: {self.best_val_metric:.4f}")


if __name__ == "__main__":
    print("Trainer module loaded")
