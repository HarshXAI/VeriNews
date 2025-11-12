"""
Enhanced GAT training with comprehensive improvements (Full-Batch Version):
- Temporal/source-grouped splits to prevent leakage
- GATv2 with residual connections
- Focal loss for class imbalance
- Threshold tuning for optimal F1
- Temperature scaling for calibration
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
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedGATv2(nn.Module):
    """Enhanced GAT with GATv2, residual connections, and LayerNorm"""
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels,
        num_layers=3,
        num_heads=6,
        dropout=0.5,
        drop_edge_p=0.2,
        use_edge_attr=False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p
        self.use_edge_attr = use_edge_attr
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATv2Conv(
                in_channels, 
                hidden_channels, 
                heads=num_heads, 
                dropout=dropout,
                edge_dim=1 if use_edge_attr else None,
                concat=True
            )
        )
        self.norms.append(LayerNorm(hidden_channels * num_heads))
        self.residual_projections.append(
            nn.Linear(in_channels, hidden_channels * num_heads)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=1 if use_edge_attr else None,
                    concat=True
                )
            )
            self.norms.append(LayerNorm(hidden_channels * num_heads))
            self.residual_projections.append(
                nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
            )
        
        # Output layer
        self.convs.append(
            GATv2Conv(
                hidden_channels * num_heads,
                out_channels,
                heads=1,
                dropout=dropout,
                edge_dim=1 if use_edge_attr else None,
                concat=False
            )
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        for i in range(self.num_layers - 1):
            # Store input for residual
            residual = self.residual_projections[i](x)
            
            # DropEdge (training only)
            if self.training and self.drop_edge_p > 0:
                mask = torch.rand(edge_index.size(1)) > self.drop_edge_p
                edge_index_dropped = edge_index[:, mask]
                edge_attr_dropped = edge_attr[mask] if edge_attr is not None else None
            else:
                edge_index_dropped = edge_index
                edge_attr_dropped = edge_attr
            
            # GAT convolution
            x = self.convs[i](x, edge_index_dropped, edge_attr=edge_attr_dropped)
            
            # Residual connection + LayerNorm
            x = self.norms[i](x + residual)
            
            # Activation + Dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no residual)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        return x


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """Fit temperature using NLL optimization"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits = logits.detach()
        
        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)


def create_temporal_splits(
    data, 
    train_ratio=0.7, 
    val_ratio=0.15,
    metadata_path=None
):
    """Create temporal train/val/test splits"""
    num_nodes = data.num_nodes
    
    if metadata_path and os.path.exists(metadata_path):
        # Use timestamps from metadata
        metadata = pd.read_csv(metadata_path)
        sorted_indices = metadata.sort_values('timestamp').index.values
    else:
        # Use node order as temporal proxy
        print("  ⚠️  No metadata, using node order as temporal proxy")
        sorted_indices = np.arange(num_nodes)
    
    # Split points
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[sorted_indices[:train_end]] = True
    val_mask[sorted_indices[train_end:val_end]] = True
    test_mask[sorted_indices[val_end:]] = True
    
    return train_mask, val_mask, test_mask


def find_optimal_threshold(probs, labels):
    """Find optimal classification threshold for F1"""
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


def train_epoch(model, data, train_mask, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    out = model(data.x, data.edge_index, edge_attr=edge_attr)
    
    # Loss on training nodes only
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Training accuracy
    pred = out[train_mask].argmax(dim=-1)
    correct = (pred == data.y[train_mask]).sum()
    acc = int(correct) / int(train_mask.sum())
    
    return float(loss), acc


@torch.no_grad()
def evaluate(model, data, mask, criterion, device, return_logits=False):
    """Evaluate on a mask"""
    model.eval()
    
    # Forward pass
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    out = model(data.x, data.edge_index, edge_attr=edge_attr)
    
    # Loss
    loss = criterion(out[mask], data.y[mask])
    
    # Predictions
    pred = out[mask].argmax(dim=-1)
    labels = data.y[mask].cpu().numpy()
    preds = pred.cpu().numpy()
    
    # Probabilities
    probs = F.softmax(out[mask], dim=-1)
    probs_pos = probs[:, 1].cpu().numpy()
    
    # Metrics
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(labels, probs_pos)
    except:
        auc = 0.0
    
    results = {
        'loss': float(loss),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'predictions': preds,
        'probabilities': probs_pos,
        'labels': labels
    }
    
    if return_logits:
        results['logits'] = out[mask]
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Enhanced GAT Training (Full-Batch)')
    parser.add_argument('--data-path', type=str, default='data/graphs/graph_data.pt')
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='experiments/improved')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--drop-edge', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--focal-loss', action='store_true', default=True)
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--use-edge-weights', action='store_true', default=False)
    parser.add_argument('--calibrate', action='store_true', default=True)
    parser.add_argument('--tune-threshold', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = "cpu" if args.device == "auto" else args.device
    
    print("=" * 70)
    print("🚀 ENHANCED GAT TRAINING (FULL-BATCH)")
    print("=" * 70)
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    data = torch.load(args.data_path, weights_only=False)
    data = data.to(device)
    
    print(f"  ✓ Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"  ✓ Features: {data.x.shape}")
    
    # Class distribution
    num_fake = (data.y == 0).sum().item()
    num_real = (data.y == 1).sum().item()
    imbalance_ratio = num_real / max(num_fake, 1)
    print(f"  ✓ Class distribution: Fake={num_fake}, Real={num_real}")
    print(f"  ⚠️  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Create splits
    print("\n📊 Creating temporal splits...")
    train_mask, val_mask, test_mask = create_temporal_splits(
        data, 
        metadata_path=args.metadata_path
    )
    
    print(f"  ✓ Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Model
    print(f"\n🧠 Initializing ImprovedGATv2...")
    model = ImprovedGATv2(
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        drop_edge_p=args.drop_edge,
        use_edge_attr=args.use_edge_weights
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  • Hidden dim: {args.hidden_dim}")
    print(f"  • Layers: {args.num_layers}")
    print(f"  • Heads: {args.num_heads}")
    print(f"  • Dropout: {args.dropout}")
    print(f"  • DropEdge: {args.drop_edge}")
    print(f"  ✓ Parameters: {num_params:,}")
    
    # Loss
    if args.focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"  ✓ Using Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  ✓ Using Cross-Entropy Loss")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, data, train_mask, optimizer, criterion, device
        )
        
        # Validate
        val_results = evaluate(model, data, val_mask, criterion, device)
        
        # Update scheduler
        scheduler.step(val_results['f1'])
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"\nEpoch {epoch:3d}/{args.epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"  Val:   Loss={val_results['loss']:.4f}, Acc={val_results['accuracy']:.4f}, "
                  f"F1={val_results['f1']:.4f}, AUC={val_results['auc']:.4f}")
        
        # Save best model
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    print(f"\n✅ Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    val_results = evaluate(model, data, val_mask, criterion, device, return_logits=True)
    test_results = evaluate(model, data, test_mask, criterion, device)
    
    # Calibration
    if args.calibrate:
        print("\n🔧 Calibrating probabilities...")
        temp_scaler = TemperatureScaling().to(device)
        temp_scaler.fit(val_results['logits'], data.y[val_mask])
        print(f"  ✓ Optimal temperature: {temp_scaler.temperature.item():.3f}")
        
        # Re-evaluate with calibration
        model.eval()
        with torch.no_grad():
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            out = model(data.x, data.edge_index, edge_attr=edge_attr)
            out_calibrated = temp_scaler(out)
            
            probs_calibrated = F.softmax(out_calibrated[test_mask], dim=-1)
            test_results['probabilities'] = probs_calibrated[:, 1].cpu().numpy()
    
    # Threshold tuning
    if args.tune_threshold:
        print("\n🎯 Tuning classification threshold...")
        optimal_threshold, tuned_f1 = find_optimal_threshold(
            val_results['probabilities'], 
            val_results['labels']
        )
        print(f"  ✓ Optimal threshold: {optimal_threshold:.3f} (F1={tuned_f1:.4f})")
        
        # Apply to test set
        test_preds_tuned = (test_results['probabilities'] >= optimal_threshold).astype(int)
        _, _, test_f1_tuned, _ = precision_recall_fscore_support(
            test_results['labels'], test_preds_tuned, average='binary', zero_division=0
        )
        print(f"  ✓ Test F1 with tuned threshold: {test_f1_tuned:.4f}")
        test_results['f1_tuned'] = test_f1_tuned
        test_results['threshold'] = optimal_threshold
    
    # Print results
    print("\n📊 Test Results:")
    print(f"  • Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  • Precision: {test_results['precision']:.4f}")
    print(f"  • Recall:    {test_results['recall']:.4f}")
    print(f"  • F1 Score:  {test_results['f1']:.4f}")
    if 'f1_tuned' in test_results:
        print(f"  • F1 (tuned): {test_results['f1_tuned']:.4f}")
    print(f"  • AUC:       {test_results['auc']:.4f}")
    
    # Save results
    results_dict = {
        'test_accuracy': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1'],
        'test_f1_tuned': test_results.get('f1_tuned', test_results['f1']),
        'test_auc': test_results['auc'],
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'threshold': test_results.get('threshold', 0.5),
        'num_parameters': num_params,
        'config': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
