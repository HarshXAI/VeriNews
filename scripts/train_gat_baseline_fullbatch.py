"""
Baseline GAT training (Full-Batch Version) for comparison
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import set_seed, get_device


class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_temporal_splits(data, train_ratio=0.7, val_ratio=0.15):
    """Create temporal train/val/test splits (same as improved version)"""
    num_nodes = data.num_nodes
    
    # Use node order as temporal proxy
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


def train_epoch(model, data, train_mask, optimizer, device):
    """Train one epoch"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    pred = out[train_mask].argmax(dim=-1)
    correct = (pred == data.y[train_mask]).sum()
    acc = int(correct) / int(train_mask.sum())
    
    return float(loss), acc


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate on a mask"""
    model.eval()
    
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[mask], data.y[mask])
    
    pred = out[mask].argmax(dim=-1)
    labels = data.y[mask].cpu().numpy()
    preds = pred.cpu().numpy()
    
    # Probabilities
    probs = torch.exp(out[mask])
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
    
    return {
        'loss': float(loss),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline GAT Training (Full-Batch)')
    parser.add_argument('--data-path', type=str, default='data/graphs/graph_data.pt')
    parser.add_argument('--output-dir', type=str, default='experiments/baseline')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = "cpu" if args.device == "auto" else args.device
    
    print("=" * 70)
    print("🚀 BASELINE GAT TRAINING (FULL-BATCH)")
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
    print(f"  ✓ Class distribution: Fake={num_fake}, Real={num_real}")
    
    # Create splits
    print("\n📊 Creating temporal splits...")
    train_mask, val_mask, test_mask = create_temporal_splits(data)
    print(f"  ✓ Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Model
    print(f"\n🧠 Initializing SimpleGAT...")
    model = SimpleGATNode(
        in_channels=data.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  • Hidden dim: {args.hidden_dim}")
    print(f"  • Heads: {args.num_heads}")
    print(f"  • Dropout: {args.dropout}")
    print(f"  ✓ Parameters: {num_params:,}")
    
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
            model, data, train_mask, optimizer, device
        )
        
        # Validate
        val_results = evaluate(model, data, val_mask, device)
        
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
    
    test_results = evaluate(model, data, test_mask, device)
    
    print("\n📊 Test Results:")
    print(f"  • Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  • Precision: {test_results['precision']:.4f}")
    print(f"  • Recall:    {test_results['recall']:.4f}")
    print(f"  • F1 Score:  {test_results['f1']:.4f}")
    print(f"  • AUC:       {test_results['auc']:.4f}")
    
    # Save results
    results_dict = {
        'test_accuracy': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1'],
        'test_auc': test_results['auc'],
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'num_parameters': num_params,
        'config': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
