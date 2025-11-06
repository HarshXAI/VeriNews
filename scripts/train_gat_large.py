"""
Week 1-2: Train GAT on full dataset with neighbor sampling
Large-scale training script for 23K articles
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device


class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv
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


def create_splits(data, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test splits"""
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    return train_mask, val_mask, test_mask


def train_epoch_minibatch(model, train_loader, optimizer, device):
    """Train one epoch with mini-batches"""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        
        # Only compute loss on target nodes (not neighbors)
        loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * batch.batch_size
        pred = out[:batch.batch_size].argmax(dim=-1)
        total_correct += int((pred == batch.y[:batch.batch_size]).sum())
        total_examples += batch.batch_size
    
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate_minibatch(model, loader, device):
    """Evaluate with mini-batches"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    total_examples = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        
        # Only evaluate target nodes
        loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        total_loss += float(loss) * batch.batch_size
        
        pred = out[:batch.batch_size].argmax(dim=-1)
        probs = torch.exp(out[:batch.batch_size])[:, 1]
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y[:batch.batch_size].cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
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
    
    return {
        'loss': total_loss / total_examples,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/full_dataset")
    parser.add_argument("--num-nodes", type=int, default=None, help="Limit number of nodes (None = all)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-neighbors", type=int, nargs='+', default=[10, 10, 5])
    parser.add_argument("--lr", type=float, default=0.001)
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
    print("🚀 LARGE-SCALE GAT TRAINING")
    print("="*70)
    
    device = get_device() if args.device == "auto" else args.device
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    data = torch.load(args.data_path, weights_only=False)
    
    # Optionally limit nodes for testing
    if args.num_nodes is not None and args.num_nodes < data.num_nodes:
        print(f"  ⚠️  Limiting to {args.num_nodes} nodes for testing")
        indices = torch.randperm(data.num_nodes)[:args.num_nodes]
        data.x = data.x[indices]
        data.y = data.y[indices]
        # Note: Would need to filter edges too for proper subset
    
    print(f"  ✓ Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"  ✓ Features: {data.x.shape}, Classes: {data.y.max().item() + 1}")
    
    # Create splits
    print(f"\n📊 Creating train/val/test splits...")
    train_mask, val_mask, test_mask = create_splits(data)
    print(f"  ✓ Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Create data loaders
    print(f"\n🔄 Creating mini-batch loaders...")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Neighbor sampling: {args.num_neighbors} (per layer)")
    
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
    
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n🧠 Initializing model...")
    model = SimpleGATNode(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=data.y.max().item() + 1,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_f1 = 0
    patience_counter = 0
    train_time = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch_minibatch(model, train_loader, optimizer, device)
        val_metrics = evaluate_minibatch(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        train_time += epoch_time
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Time: {epoch_time:.1f}s | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.3f} | "
                  f"Val: {val_metrics['accuracy']:.3f} | "
                  f"F1: {val_metrics['f1']:.3f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\n⏸️  Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("📈 FINAL EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_metrics = evaluate_minibatch(model, train_loader, device)
    val_metrics = evaluate_minibatch(model, val_loader, device)
    test_metrics = evaluate_minibatch(model, test_loader, device)
    
    print(f"\n📊 Results (Best Epoch: {checkpoint['epoch']}):")
    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
    print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
    print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    print(f"\n⏱️  Total training time: {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"  Average time per epoch: {train_time/checkpoint['epoch']:.1f}s")
    
    # Save results
    results = {
        'model_params': num_params,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'best_epoch': checkpoint['epoch'],
        'training_time_seconds': train_time,
        'train_metrics': {k: float(v) for k, v in train_metrics.items()},
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'hyperparameters': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 Test Results:")
    print(f"  • Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  • F1-Score: {test_metrics['f1']:.4f}")
    print(f"  • AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"\n📁 Saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
