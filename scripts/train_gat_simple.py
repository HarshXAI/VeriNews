"""
Train GAT for node-level fake news detection
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device


class SimpleGATNode(nn.Module):
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


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """Split data"""
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    labels = data.y.cpu().numpy()
    
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=labels, random_state=42
    )
    
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio_adjusted, 
        stratify=labels[temp_idx], random_state=42
    )
    
    return train_idx, val_idx, test_idx


def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    pred = out[train_mask].argmax(dim=1)
    acc = (pred == data.y[train_mask]).float().mean()
    return loss.item(), acc.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = torch.exp(out[mask])[:, 1].cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    loss = F.nll_loss(out[mask], data.y[mask]).item()
    
    return {
        'loss': loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🚀 TRAINING GAT FOR FAKE NEWS DETECTION")
    print("="*70)
    
    device = get_device() if args.device == "auto" else args.device
    print(f"\n🖥️  Device: {device}")
    
    print(f"\n📂 Loading data...")
    data = torch.load(args.data_path, weights_only=False).to(device)
    
    print(f"  ✓ Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"  ✓ Features: {data.x.shape}, Classes: {data.y.max().item() + 1}")
    
    # Split
    train_idx, val_idx, test_idx = split_data(data)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"\n📊 Split: Train {train_mask.sum()}, Val {val_mask.sum()}, Test {test_mask.sum()}")
    
    # Model
    model = SimpleGATNode(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=data.y.max().item() + 1,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {num_params:,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, data, optimizer, train_mask)
        val_metrics = evaluate(model, data, val_mask)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} "
                  f"Train: {train_acc:.3f} Val: {val_metrics['accuracy']:.3f} "
                  f"F1: {val_metrics['f1']:.3f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_f1': best_val_f1,
            }, os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\n⏸️  Early stop at epoch {epoch}")
            break
    
    # Eval
    print("\n" + "="*70)
    print("📈 FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_metrics = evaluate(model, data, train_mask)
    val_metrics = evaluate(model, data, val_mask)
    test_metrics = evaluate(model, data, test_mask)
    
    print("\n📊 Results:")
    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    # Save
    results = {
        'model_params': num_params,
        'best_epoch': checkpoint['epoch'],
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n�� Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"🎯 Test F1: {test_metrics['f1']:.4f}")
    print(f"🎯 Test AUC: {test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
