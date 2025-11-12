"""
Graph Transformer with Class Weights - Final Push to 92%
=========================================================

Add class weights to boost Class 0 performance (currently 75.23%)

Strategy: Weight Class 0 by 2.5x (it's ~1/3 the size, needs more emphasis)

Current: 91.94% F1, Class 0: 75.23%
Target: 92.0%+ F1, Class 0: 77-78%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device('cpu')
print(f"Using device: {device}")

output_dir = Path("experiments/graph_transformer_weighted")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GRAPH TRANSFORMER WITH CLASS WEIGHTS")
print("=" * 80)
print()

# Load data
print("1. Loading data...")
data = torch.load('data/graphs_full/graph_data_with_node2vec.pt', weights_only=False)
splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)

train_mask = splits['train_mask']
val_mask = splits['val_mask']
test_mask = splits['test_mask']

data = data.to(device)

print(f"Features: {data.x.shape[1]}")
print(f"Nodes: {data.num_nodes}")
print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

# Calculate class distribution
train_labels = data.y[train_mask]
class_0_count = (train_labels == 0).sum().item()
class_1_count = (train_labels == 1).sum().item()
total = class_0_count + class_1_count

print(f"\nClass Distribution:")
print(f"   Class 0: {class_0_count} ({class_0_count/total*100:.1f}%)")
print(f"   Class 1: {class_1_count} ({class_1_count/total*100:.1f}%)")
print(f"   Ratio:   1:{class_1_count/class_0_count:.2f}")
print()

# Class weights (boost minority class)
class_weights = torch.tensor([2.5, 1.0], dtype=torch.float32).to(device)
print(f"Class Weights: {class_weights.tolist()}")
print()

# Model definition
class GraphTransformerLayer(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.2):
        super().__init__()
        
        self.gat = GATv2Conv(channels, channels, heads=num_heads, 
                            dropout=dropout, concat=False, add_self_loops=True)
        
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, 
                                         batch_first=True)
        
        self.vn_update = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        
        self.node_update = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, virtual_node):
        x_local = self.gat(x, edge_index)
        x = self.norm1(x + F.dropout(x_local, p=self.dropout, training=self.training))
        
        vn_input = x.mean(dim=0, keepdim=True)
        vn_update = self.vn_update(vn_input)
        virtual_node = self.norm2(virtual_node + vn_update)
        
        vn_broadcast = virtual_node.expand(x.size(0), -1)
        x_combined = torch.cat([x, vn_broadcast], dim=1)
        x_update = self.node_update(x_combined)
        x = self.norm3(x + x_update)
        
        return x, virtual_node

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=4, 
                 num_heads=8, dropout=0.2, num_classes=2):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_channels))
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        vn = self.virtual_node
        
        for layer in self.layers:
            x, vn = layer(x, edge_index, vn)
        
        x = self.output(x)
        return x

# Create model
print("=" * 80)
print("TRAINING")
print("=" * 80)
print()

torch.manual_seed(42)
np.random.seed(42)

model = GraphTransformer(
    in_channels=data.x.shape[1],
    hidden_channels=256,
    num_layers=4,
    num_heads=8,
    dropout=0.2
).to(device)

print(f"Model: Graph Transformer with Class Weights (seed 42)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

# Training loop
best_val_f1 = 0
best_model_state = None
patience_counter = 0
patience = 20
epochs = 200

pbar = tqdm(range(epochs), desc="Training")

for epoch in pbar:
    # Training with class weights
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    
    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[val_mask].argmax(dim=1)
            val_f1 = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='weighted')
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'val_f1': f'{val_f1:.4f}',
                'best': f'{best_val_f1:.4f}'
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience // 10:
                print(f"\n✅ Early stopped at epoch {epoch+1}")
                break

pbar.close()

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\n✅ Training complete!")
print(f"   Best validation F1: {best_val_f1:.4f}")

# Evaluate
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)
print()

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=1)

test_f1 = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='weighted')
test_acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())

cm = confusion_matrix(data.y[test_mask].cpu(), pred.cpu())
class_0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

print(f"📊 Overall Performance:")
print(f"   Test F1:       {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print()
print(f"📊 Confusion Matrix:")
print(f"           Predicted")
print(f"           0      1")
print(f"Actual 0   {cm[0,0]:<6} {cm[0,1]:<6}  ({class_0_recall*100:.1f}% recall)")
print(f"       1   {cm[1,0]:<6} {cm[1,1]:<6}  ({class_1_recall*100:.1f}% recall)")
print()
print(f"📊 Per-Class Performance:")
print(f"   Class 0 recall: {class_0_recall:.4f} ({class_0_recall*100:.2f}%)")
print(f"   Class 1 recall: {class_1_recall:.4f} ({class_1_recall*100:.2f}%)")
print()

# Comparison
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

baseline_gt = 0.9194
baseline_class0 = 0.7523

print(f"Graph Transformer (no weights): {baseline_gt:.4f}, Class 0: {baseline_class0:.4f}")
print(f"Graph Transformer (weighted):   {test_f1:.4f}, Class 0: {class_0_recall:.4f}")
print()
print(f"F1 Change:      {test_f1 - baseline_gt:+.4f} ({(test_f1 - baseline_gt)*100:+.2f} pts)")
print(f"Class 0 Change: {class_0_recall - baseline_class0:+.4f} ({(class_0_recall - baseline_class0)*100:+.2f} pts)")
print()

# Save results
results = {
    'test_f1': float(test_f1),
    'test_acc': float(test_acc),
    'val_f1': float(best_val_f1),
    'class_0_recall': float(class_0_recall),
    'class_1_recall': float(class_1_recall),
    'confusion_matrix': cm.tolist(),
    'improvement_vs_unweighted': float(test_f1 - baseline_gt),
    'class_0_improvement': float(class_0_recall - baseline_class0),
    'class_weights': class_weights.cpu().tolist()
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to {output_dir / 'results.json'}")
print()

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

target = 0.92
print(f"🎯 Target: {target:.4f} (92.00%)")
print(f"   Achieved: {test_f1:.4f} ({test_f1*100:.2f}%)")
print()

if test_f1 >= target:
    print("🎉🎉🎉 SUCCESS! 92% TARGET REACHED! 🎉🎉🎉")
    print()
    print(f"   Total journey: 86.61% → {test_f1*100:.2f}% = +{(test_f1-0.8661)*100:.2f} pts")
    print(f"   Gap to 95%: {(0.95-test_f1)*100:.2f} pts remaining")
else:
    gap = target - test_f1
    print(f"✅ Very close! Gap: {gap:.4f} ({gap*100:.2f} pts)")

print()
print("=" * 80)
