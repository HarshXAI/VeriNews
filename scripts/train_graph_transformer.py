"""
Efficient Graph Transformer - Memory-Optimized Version
======================================================

Instead of full GraphGPS with global attention on 23K nodes (memory intensive),
we use a hybrid approach:

1. **Graph Attention (GAT)** - Local neighborhood attention (efficient)
2. **Virtual Node** - Global communication via a learnable super-node
3. **Deeper architecture** - More layers to compensate for no global attention

This is inspired by GraphGPS but optimized for large graphs on CPU.

Current: 91.49% F1
Target: 92.0-92.5% F1 (+0.5-1.0 pts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device('cpu')
print(f"Using device: {device}")

output_dir = Path("experiments/graph_transformer")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GRAPH TRANSFORMER WITH VIRTUAL NODE")
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

print(f"Features: {data.x.shape[1]} (394 baseline + 128 Node2Vec)")
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
print()

class GraphTransformerLayer(nn.Module):
    """Efficient graph layer with virtual node for global context"""
    def __init__(self, channels, num_heads=8, dropout=0.2):
        super().__init__()
        
        # Local graph attention
        self.gat = GATv2Conv(channels, channels, heads=num_heads, 
                            dropout=dropout, concat=False, add_self_loops=True)
        
        # Virtual node MLP (for global information)
        self.vn_update = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        
        # Node update with virtual node
        self.node_update = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, virtual_node):
        """
        x: [num_nodes, channels]
        edge_index: [2, num_edges]
        virtual_node: [1, channels]
        """
        # 1. Local graph attention
        x_local = self.gat(x, edge_index)
        x = self.norm1(x + F.dropout(x_local, p=self.dropout, training=self.training))
        
        # 2. Update virtual node (aggregate from all nodes)
        vn_input = x.mean(dim=0, keepdim=True)  # Global average
        vn_update = self.vn_update(vn_input)
        virtual_node = self.norm2(virtual_node + vn_update)
        
        # 3. Update nodes with virtual node (broadcast global context)
        vn_broadcast = virtual_node.expand(x.size(0), -1)
        x_combined = torch.cat([x, vn_broadcast], dim=1)
        x_update = self.node_update(x_combined)
        x = self.norm3(x + x_update)
        
        return x, virtual_node

class GraphTransformer(nn.Module):
    """Graph Transformer with Virtual Node - Memory Efficient"""
    def __init__(self, in_channels, hidden_channels=256, num_layers=4, 
                 num_heads=8, dropout=0.2, num_classes=2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Virtual node embedding (learnable global context)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_channels))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        
    def forward(self, x, edge_index):
        # Project input
        x = self.input_proj(x)
        
        # Initialize virtual node for this forward pass
        vn = self.virtual_node
        
        # Apply transformer layers
        for layer in self.layers:
            x, vn = layer(x, edge_index, vn)
        
        # Output
        x = self.output(x)
        return x

# Create model
print("=" * 80)
print("CREATING MODEL")
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

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: Graph Transformer with Virtual Node (seed 42)")
print(f"Parameters: {num_params:,}")
print(f"Hidden channels: 256")
print(f"Num layers: 4")
print(f"Num heads: 8")
print()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

# Training loop
print("=" * 80)
print("TRAINING")
print("=" * 80)
print()

best_val_f1 = 0
best_model_state = None
patience_counter = 0
patience = 20
epochs = 200

pbar = tqdm(range(epochs), desc="Training")

for epoch in pbar:
    # Training
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation every 10 epochs
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

# Evaluate on test set
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)
print()

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=1)
    probs = F.softmax(out[test_mask], dim=1)

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
print("COMPARISON WITH BASELINES")
print("=" * 80)
print()

baseline_ensemble = 0.9149
baseline_single = 0.9113

print(f"Baseline (Node2Vec ensemble): {baseline_ensemble:.4f}")
print(f"Baseline (Node2Vec single):   {baseline_single:.4f}")
print(f"Current (Graph Transformer):  {test_f1:.4f}")
print()
print(f"Improvement vs ensemble: {test_f1 - baseline_ensemble:+.4f} ({(test_f1 - baseline_ensemble)*100:+.2f} pts)")
print(f"Improvement vs single:   {test_f1 - baseline_single:+.4f} ({(test_f1 - baseline_single)*100:+.2f} pts)")
print()

baseline_class0 = 0.7581
print(f"Class 0 Recall:")
print(f"   Baseline: {baseline_class0:.4f} ({baseline_class0*100:.2f}%)")
print(f"   Current:  {class_0_recall:.4f} ({class_0_recall*100:.2f}%)")
print(f"   Change:   {class_0_recall - baseline_class0:+.4f} ({(class_0_recall - baseline_class0)*100:+.2f} pts)")
print()

# Save results
results = {
    'test_f1': float(test_f1),
    'test_acc': float(test_acc),
    'val_f1': float(best_val_f1),
    'class_0_recall': float(class_0_recall),
    'class_1_recall': float(class_1_recall),
    'confusion_matrix': cm.tolist(),
    'improvement_vs_ensemble': float(test_f1 - baseline_ensemble),
    'improvement_vs_single': float(test_f1 - baseline_single),
    'class_0_improvement': float(class_0_recall - baseline_class0),
    'model_params': num_params,
    'architecture': 'Graph Transformer with Virtual Node',
    'hyperparameters': {
        'hidden_channels': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.2,
        'lr': 0.001,
        'weight_decay': 5e-4
    }
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

target_f1 = 0.92
print(f"🎯 Target: {target_f1:.4f} (92.00%)")
print(f"   Achieved: {test_f1:.4f} ({test_f1*100:.2f}%)")
print()

if test_f1 >= target_f1:
    print("🎉 SUCCESS! Target reached!")
    gap_to_95 = 0.95 - test_f1
    print(f"   Gap to 95%: {gap_to_95:.4f} ({gap_to_95*100:.2f} pts)")
elif test_f1 >= baseline_ensemble:
    print("✅ Improvement over baseline!")
    gap = target_f1 - test_f1
    print(f"   Gap to 92%: {gap:.4f} ({gap*100:.2f} pts)")
else:
    print("⚠️  No improvement over baseline")
    print(f"   Baseline was better by {baseline_ensemble - test_f1:.4f} pts")

print()
print(f"📊 Key Metrics:")
print(f"   Overall F1:      {test_f1:.4f}")
print(f"   Class 0 recall:  {class_0_recall:.4f} (baseline: {baseline_class0:.4f})")
print(f"   Class 1 recall:  {class_1_recall:.4f}")
print(f"   Model params:    {num_params:,}")
print()
print("=" * 80)
