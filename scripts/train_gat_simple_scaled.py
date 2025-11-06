"""
Week 1-2: Train GAT on larger dataset (simplified approach without neighbor sampling)
Works well for datasets up to ~10K nodes on MPS
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device


class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test splits"""
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def train_epoch(model, data, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask].to(device)).sum().item() / data.train_mask.sum().item()
    
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, device, mask):
    """Evaluate model"""
    model.eval()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out[mask].argmax(dim=1).cpu()
    y_true = data.y[mask].cpu()
    
    # Metrics
    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    
    # AUC
    try:
        probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_true, probs)
    except:
        auc = 0.0
    
    loss = F.cross_entropy(out[mask], y_true.to(device)).item()
    
    return loss, acc, prec, rec, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num-nodes", type=int, default=None, help="Limit to first N nodes")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 SCALED GAT TRAINING")
    print("="*70)
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data from {args.data_path}...")
    data = torch.load(args.data_path, weights_only=False)
    
    # Limit nodes if specified
    if args.num_nodes and args.num_nodes < data.x.shape[0]:
        print(f"  📉 Limiting to first {args.num_nodes} nodes...")
        data.x = data.x[:args.num_nodes]
        data.y = data.y[:args.num_nodes]
        
        # Filter edges
        mask = (data.edge_index[0] < args.num_nodes) & (data.edge_index[1] < args.num_nodes)
        data.edge_index = data.edge_index[:, mask]
    
    print(f"  ✓ Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    print(f"  ✓ Features: {data.x.shape}, Classes: {data.y.max().item() + 1}")
    
    # Create splits
    print(f"\n📊 Creating train/val/test splits...")
    data = split_data(data)
    print(f"  ✓ Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = SimpleGATNode(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, data, device, data.val_mask)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'gat_model_best_scaled.pt'))
            print(f"  💾 Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'gat_model_best_scaled.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(model, data, device, data.test_mask)
    
    print(f"\n📊 Test Set Performance:")
    print(f"  • Accuracy:  {test_acc:.4f}")
    print(f"  • Precision: {test_prec:.4f}")
    print(f"  • Recall:    {test_rec:.4f}")
    print(f"  • F1-Score:  {test_f1:.4f}")
    print(f"  • AUC:       {test_auc:.4f}")
    
    # Save metrics
    metrics = {
        'training_time': training_time,
        'best_epoch': checkpoint['epoch'],
        'num_params': num_params,
        'num_nodes': data.x.shape[0],
        'num_edges': data.edge_index.shape[1],
        'test_metrics': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1': float(test_f1),
            'auc': float(test_auc)
        },
        'val_metrics': {
            'accuracy': float(checkpoint['val_acc']),
            'f1': float(checkpoint['val_f1'])
        },
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'config': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'training_metrics_scaled.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n⏱️  Training time: {training_time:.2f}s ({training_time/60:.2f}m)")
    print(f"💾 Model saved to: {args.output_dir}/gat_model_best_scaled.pt")
    print(f"📊 Metrics saved to: {args.output_dir}/training_metrics_scaled.json")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
