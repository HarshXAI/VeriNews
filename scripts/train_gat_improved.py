"""
Enhanced GAT training with comprehensive improvements:
- Temporal/source-grouped splits to prevent leakage
- GATv2 with edge weights and residual connections
- Focal loss for class imbalance
- Threshold tuning for optimal F1
- Temperature scaling for calibration
- Hyperparameter optimization support
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv, LayerNorm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, roc_curve
)
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import set_seed, get_device


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iters=50):
        """Fit temperature using NLL on validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)


class ImprovedGATv2(nn.Module):
    """
    Enhanced GAT with:
    - GATv2Conv (fixes static attention)
    - Residual connections
    - Layer normalization
    - Edge attribute support
    - DropEdge regularization
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.5,
        edge_dim: int = None,
        use_residual: bool = True,
        drop_edge_p: float = 0.2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.drop_edge_p = drop_edge_p
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers
        for i in range(num_layers):
            # First layer
            if i == 0:
                conv = GATv2Conv(
                    hidden_channels,
                    hidden_channels,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    share_weights=False
                )
                skip = nn.Linear(hidden_channels, hidden_channels * num_heads)
                norm = LayerNorm(hidden_channels * num_heads)
            # Last layer
            elif i == num_layers - 1:
                conv = GATv2Conv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    share_weights=False
                )
                skip = nn.Linear(hidden_channels * num_heads, hidden_channels)
                norm = LayerNorm(hidden_channels)
            # Middle layers
            else:
                conv = GATv2Conv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    share_weights=False
                )
                skip = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
                norm = LayerNorm(hidden_channels * num_heads)
            
            self.convs.append(conv)
            self.norms.append(norm)
            self.skips.append(skip)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def drop_edge(self, edge_index, edge_attr=None, p=0.5, training=True):
        """DropEdge regularization"""
        if not training or p == 0:
            return edge_index, edge_attr
        
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > p
        edge_index = edge_index[:, mask]
        
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        
        return edge_index, edge_attr
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_channels]
            edge_index: [2, E]
            edge_attr: [E, edge_dim] optional edge features
        """
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)
        
        # Apply DropEdge
        edge_index, edge_attr = self.drop_edge(
            edge_index, edge_attr, 
            p=self.drop_edge_p, 
            training=self.training
        )
        
        # GAT layers with residual connections
        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skips)):
            x_skip = skip(x) if self.use_residual else 0
            
            # GAT convolution
            x = conv(x, edge_index, edge_attr=edge_attr)
            
            # Residual + norm + activation
            if self.use_residual:
                x = x + x_skip
            x = norm(x)
            
            # Activation (except last layer)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        return self.classifier(x)


def create_temporal_splits(
    data,
    timestamps: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create temporal train/val/test splits to prevent leakage.
    Train on earlier data, validate/test on later data.
    """
    sorted_indices = np.argsort(timestamps)
    num_nodes = len(timestamps)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[sorted_indices[:train_size]] = True
    val_mask[sorted_indices[train_size:train_size + val_size]] = True
    test_mask[sorted_indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask


def create_source_grouped_splits(
    data,
    source_ids: np.ndarray,
    n_splits: int = 5,
    val_fold: int = 0,
    test_fold: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create source-grouped K-fold splits to prevent leakage.
    Articles from the same source never appear in both train and test.
    """
    num_nodes = len(source_ids)
    gkf = GroupKFold(n_splits=n_splits)
    
    # Create dummy y for split generation
    y_dummy = np.zeros(num_nodes)
    
    folds = list(gkf.split(y_dummy, y_dummy, groups=source_ids))
    
    # Use specified folds
    val_indices = folds[val_fold][1]
    test_indices = folds[test_fold][1]
    train_indices = np.array([i for i in range(num_nodes) 
                             if i not in val_indices and i not in test_indices])
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal classification threshold for F1 score.
    
    Returns:
        optimal_threshold: Best threshold
        best_f1: F1 score at optimal threshold
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
    
    return optimal_threshold, best_f1


def train_epoch_minibatch(
    model, 
    train_loader, 
    optimizer, 
    device,
    criterion,
    use_edge_attr: bool = False
):
    """Train one epoch with mini-batches"""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        edge_attr = batch.edge_attr if use_edge_attr and hasattr(batch, 'edge_attr') else None
        out = model(batch.x, batch.edge_index, edge_attr=edge_attr)
        
        # Loss only on target nodes
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += float(loss) * batch.batch_size
        pred = out[:batch.batch_size].argmax(dim=-1)
        total_correct += int((pred == batch.y[:batch.batch_size]).sum())
        total_examples += batch.batch_size
    
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate_minibatch(
    model, 
    loader, 
    device,
    criterion,
    use_edge_attr: bool = False,
    return_logits: bool = False
):
    """Evaluate with mini-batches"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []
    total_loss = 0
    total_examples = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        edge_attr = batch.edge_attr if use_edge_attr and hasattr(batch, 'edge_attr') else None
        out = model(batch.x, batch.edge_index, edge_attr=edge_attr)
        
        # Loss on target nodes
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        total_loss += float(loss) * batch.batch_size
        
        # Predictions
        probs = F.softmax(out[:batch.batch_size], dim=-1)
        pred = probs.argmax(dim=-1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y[:batch.batch_size].cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        
        if return_logits:
            all_logits.extend(out[:batch.batch_size].cpu().numpy())
        
        total_examples += batch.batch_size
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    result = {
        'loss': total_loss / total_examples,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'probs': all_probs,
        'labels': all_labels
    }
    
    if return_logits:
        result['logits'] = np.array(all_logits)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Enhanced GAT Training')
    
    # Data
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--metadata-path", type=str, default=None,
                       help="CSV with timestamps and source_ids for grouped splits")
    parser.add_argument("--output-dir", type=str, default="experiments/improved_gat")
    
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--drop-edge-p", type=float, default=0.2)
    parser.add_argument("--use-edge-attr", action='store_true',
                       help="Use edge attributes if available")
    
    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-neighbors", type=int, nargs='+', default=[15, 10, 5])
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    
    # Loss & optimization
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    
    # Data splits
    parser.add_argument("--split-method", type=str, default="temporal",
                       choices=["random", "temporal", "source_grouped"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    
    # Calibration
    parser.add_argument("--calibrate", action='store_true',
                       help="Apply temperature scaling")
    parser.add_argument("--tune-threshold", action='store_true',
                       help="Find optimal F1 threshold on validation set")
    
    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-predictions", action='store_true')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🚀 ENHANCED GAT TRAINING")
    print("="*70)
    
    # Force CPU to avoid MPS/torch-sparse compatibility issues
    device = "cpu" if args.device == "auto" else args.device
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    data = torch.load(args.data_path, weights_only=False)
    
    print(f"  ✓ Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    print(f"  ✓ Features: {data.x.shape}")
    
    # Check class balance
    class_counts = torch.bincount(data.y)
    print(f"  ✓ Class distribution: Fake={class_counts[0]}, Real={class_counts[1]}")
    imbalance_ratio = class_counts.max().item() / class_counts.min().item()
    print(f"  ⚠️  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Create splits based on method
    print(f"\n📊 Creating {args.split_method} splits...")
    
    if args.split_method == "temporal":
        if args.metadata_path and os.path.exists(args.metadata_path):
            metadata = pd.read_csv(args.metadata_path)
            timestamps = metadata['timestamp'].values
            print(f"  ✓ Using timestamps from {args.metadata_path}")
        else:
            # Fallback: use node indices as proxy timestamps
            print("  ⚠️  No metadata, using node order as temporal proxy")
            timestamps = np.arange(data.num_nodes)
        
        train_mask, val_mask, test_mask = create_temporal_splits(
            data, timestamps, args.train_ratio, args.val_ratio
        )
    
    elif args.split_method == "source_grouped":
        if args.metadata_path and os.path.exists(args.metadata_path):
            metadata = pd.read_csv(args.metadata_path)
            source_ids = metadata['source_id'].values
            print(f"  ✓ Using source IDs from {args.metadata_path}")
        else:
            print("  ❌ Source-grouped split requires --metadata-path")
            return
        
        train_mask, val_mask, test_mask = create_source_grouped_splits(
            data, source_ids
        )
    
    else:  # random
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(args.train_ratio * num_nodes)
        val_size = int(args.val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
    
    print(f"  ✓ Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")
    
    # Create loaders
    print(f"\n🔄 Creating mini-batch loaders...")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Neighbor sampling: {args.num_neighbors}")
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=train_mask,
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=val_mask,
        shuffle=False,
    )
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=test_mask,
        shuffle=False,
    )
    
    # Create model
    print(f"\n🧠 Initializing ImprovedGATv2...")
    print(f"  • Hidden dim: {args.hidden_dim}")
    print(f"  • Layers: {args.num_layers}")
    print(f"  • Heads: {args.num_heads}")
    print(f"  • Dropout: {args.dropout}")
    print(f"  • DropEdge: {args.drop_edge_p}")
    
    edge_dim = None
    if args.use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_dim = data.edge_attr.shape[1]
        print(f"  • Edge dim: {edge_dim}")
    
    model = ImprovedGATv2(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        edge_dim=edge_dim,
        use_residual=True,
        drop_edge_p=args.drop_edge_p
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {num_params:,}")
    
    # Loss function
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"  ✓ Using Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  ✓ Using Cross-Entropy Loss")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_f1 = 0
    patience_counter = 0
    train_time = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch_minibatch(
            model, train_loader, optimizer, device, criterion, args.use_edge_attr
        )
        
        # Validate
        val_metrics = evaluate_minibatch(
            model, val_loader, device, criterion, args.use_edge_attr
        )
        
        epoch_time = time.time() - start_time
        train_time += epoch_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Time: {epoch_time:.1f}s | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'args': vars(args),
                'history': history
            }, os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n⏸️  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    print("\n" + "="*70)
    print("📈 FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(
        os.path.join(args.output_dir, 'best_model.pt'),
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get validation predictions for threshold tuning and calibration
    val_metrics = evaluate_minibatch(
        model, val_loader, device, criterion, args.use_edge_attr, return_logits=True
    )
    
    # Threshold tuning
    optimal_threshold = 0.5
    if args.tune_threshold:
        print(f"\n🎯 Tuning classification threshold...")
        optimal_threshold, tuned_f1 = find_optimal_threshold(
            val_metrics['probs'], val_metrics['labels']
        )
        print(f"  ✓ Optimal threshold: {optimal_threshold:.3f}")
        print(f"  ✓ Tuned F1: {tuned_f1:.4f} (vs {val_metrics['f1']:.4f} at 0.5)")
    
    # Temperature scaling
    temp_model = None
    if args.calibrate:
        print(f"\n🌡️  Calibrating with temperature scaling...")
        temp_model = TemperatureScaling().to(device)
        
        val_logits = torch.tensor(val_metrics['logits'], device=device)
        val_labels = torch.tensor(val_metrics['labels'], device=device)
        
        temp_model.fit(val_logits, val_labels)
        print(f"  ✓ Learned temperature: {temp_model.temperature.item():.3f}")
    
    # Final evaluation on all splits
    train_metrics = evaluate_minibatch(
        model, train_loader, device, criterion, args.use_edge_attr
    )
    test_metrics = evaluate_minibatch(
        model, test_loader, device, criterion, args.use_edge_attr
    )
    
    # Apply optimal threshold to test set
    if args.tune_threshold:
        test_preds_tuned = (test_metrics['probs'] >= optimal_threshold).astype(int)
        _, _, f1_tuned, _ = precision_recall_fscore_support(
            test_metrics['labels'], test_preds_tuned, 
            average='binary', zero_division=0
        )
        print(f"\n  Test F1 with tuned threshold: {f1_tuned:.4f}")
    
    print(f"\n📊 Results (Best Epoch: {checkpoint['epoch']}):")
    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, "
          f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
    print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, "
          f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
    print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, "
          f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    print(f"\n⏱️  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")
    
    # Save results
    results = {
        'model_params': num_params,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'class_imbalance_ratio': imbalance_ratio,
        'best_epoch': checkpoint['epoch'],
        'training_time_seconds': train_time,
        'optimal_threshold': optimal_threshold,
        'temperature': temp_model.temperature.item() if temp_model else 1.0,
        'train_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                         for k, v in train_metrics.items()},
        'val_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                        for k, v in test_metrics.items()},
        'hyperparameters': vars(args),
        'history': history
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions if requested
    if args.save_predictions:
        predictions = {
            'test_labels': test_metrics['labels'].tolist(),
            'test_probs': test_metrics['probs'].tolist(),
            'optimal_threshold': optimal_threshold
        }
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 Test Results:")
    print(f"  • Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  • F1-Score: {test_metrics['f1']:.4f}")
    print(f"  • AUC-ROC: {test_metrics['auc']:.4f}")
    if args.tune_threshold:
        print(f"  • F1 (tuned): {f1_tuned:.4f}")
    print(f"\n📁 Saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
