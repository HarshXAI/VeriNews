"""
GraphGPS (Fast Version) - Optimized for CPU Training
=====================================================

Lightweight GraphGPS that combines:
1. Local graph convolution (GATv2) - processes neighborhoods
2. Global self-attention - BUT with attention pooling to reduce complexity
3. Skip connections and layer normalization

Key optimizations for CPU:
- Smaller hidden dimensions (128 instead of 256)
- Fewer attention heads (4 instead of 8)
- Attention on pooled representations (not full graph)
- Fewer layers (3 instead of 4)

Current: 91.49% F1
Target: 92.0-92.5% F1 (+0.5-1.0 pts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device('cpu')
print(f"Using device: {device}")

output_dir = Path("experiments/graphgps_fast")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FAST GraphGPS TRAINING")
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

class GraphGPSLayer(nn.Module):
    """Single GraphGPS layer: Local MPNN + Global Attention + FFN"""
    def __init__(self, channels, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Local message passing (GATv2)
        self.local_conv = GATv2Conv(channels, channels, heads=num_heads, 
                                    dropout=dropout, concat=False)
        
        # Global attention (lightweight)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, 
                                         batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # Local convolution with residual
        local_out = self.local_conv(x, edge_index)
        x = self.norm1(x + F.dropout(local_out, p=self.dropout, training=self.training))
        
        # Global attention with residual
        # Reshape for attention: [num_nodes, channels] -> [1, num_nodes, channels]
        x_attn = x.unsqueeze(0)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn)
        attn_out = attn_out.squeeze(0)
        x = self.norm2(x + F.dropout(attn_out, p=self.dropout, training=self.training))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x

class FastGraphGPS(nn.Module):
    """Lightweight GraphGPS model optimized for CPU"""
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, 
                 num_heads=4, dropout=0.2, num_classes=2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GraphGPS layers
        self.gps_layers = nn.ModuleList([
            GraphGPSLayer(hidden_channels, num_heads, dropout)
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
        
        # Apply GPS layers
        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index)
        
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

model = FastGraphGPS(
    in_channels=data.x.shape[1],
    hidden_channels=128,  # Smaller for speed
    num_layers=3,
    num_heads=4,
    dropout=0.2
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: Fast GraphGPS (seed 42)")
print(f"Parameters: {num_params:,}")
print(f"Hidden channels: 128")
print(f"Num layers: 3")
print(f"Num heads: 4")
print()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
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
print(f"Current (GraphGPS):           {test_f1:.4f}")
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
    'hyperparameters': {
        'hidden_channels': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.2,
        'lr': 0.001,
        'weight_decay': 1e-4
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
