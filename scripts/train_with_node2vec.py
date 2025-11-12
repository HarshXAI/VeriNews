"""
Train Model with Node2Vec Embeddings
====================================

Test if Node2Vec embeddings help with hub node classification.

Current: 91.26% F1 (ensemble)
Target: 92.0-92.5% F1 (+0.7-1.2 pts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device('cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = Path("experiments/node2vec")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRAINING WITH NODE2VEC EMBEDDINGS (522 features)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data with Node2Vec embeddings...")

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

# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class GATv2Model(nn.Module):
    """GATv2 with best hyperparameters"""
    def __init__(self, in_channels, hidden_channels=256, num_heads=10, 
                 num_layers=3, dropout=0.25, num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, 
                                       heads=num_heads, dropout=dropout))
        
        self.convs.append(GATv2Conv(hidden_channels * num_heads, num_classes, 
                                   heads=1, concat=False, dropout=dropout))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================================
# 3. TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING MODEL")
print("=" * 80)

# Best hyperparameters from HPO
torch.manual_seed(42)  # Using seed 42 (our best)
np.random.seed(42)

model = GATv2Model(
    in_channels=data.x.shape[1],
    hidden_channels=256,
    num_heads=10,
    num_layers=3,
    dropout=0.25
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0016,
    weight_decay=0.000134
)

print(f"\nModel: GATv2 (seed 42)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
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
    
    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[val_mask].argmax(dim=1)
            val_f1 = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='weighted')
            val_acc = accuracy_score(data.y[val_mask].cpu(), pred.cpu())
            
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
                pbar.set_description("Training [Early stopped]")
                break

pbar.close()

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\n✅ Training complete!")
print(f"   Best validation F1: {best_val_f1:.4f}")

# ============================================================================
# 4. EVALUATE ON TEST SET
# ============================================================================
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=1)
    probs = F.softmax(out[test_mask], dim=1)

test_f1 = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='weighted')
test_acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())

print(f"\n📊 Overall Performance:")
print(f"   Test F1:       {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Per-class analysis
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(data.y[test_mask].cpu(), pred.cpu())
print(f"\n📊 Confusion Matrix:")
print(f"           Predicted")
print(f"           0      1")
print(f"Actual 0  {cm[0, 0]:4d}  {cm[0, 1]:4d}   ({cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}% recall)")
print(f"       1  {cm[1, 0]:4d}  {cm[1, 1]:4d}   ({cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}% recall)")

class0_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
class1_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print(f"\n📊 Per-Class Performance:")
print(f"   Class 0 recall: {class0_recall:.4f} ({class0_recall*100:.2f}%)")
print(f"   Class 1 recall: {class1_recall:.4f} ({class1_recall*100:.2f}%)")

# ============================================================================
# 5. COMPARE WITH BASELINE
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON WITH BASELINE")
print("=" * 80)

baseline_single = 0.9052  # GATv2 without Node2Vec (394 features)
baseline_ensemble = 0.9126  # Ensemble of 3 models

improvement_vs_single = test_f1 - baseline_single
improvement_vs_ensemble = test_f1 - baseline_ensemble

print(f"\nBaseline (394 features, single model): {baseline_single:.4f}")
print(f"Baseline (394 features, ensemble):     {baseline_ensemble:.4f}")
print(f"Current (522 features, single model):  {test_f1:.4f}")
print(f"\nImprovement vs single:   {improvement_vs_single:+.4f} ({improvement_vs_single*100:+.2f} pts)")
print(f"Improvement vs ensemble: {improvement_vs_ensemble:+.4f} ({improvement_vs_ensemble*100:+.2f} pts)")

# Class 0 comparison
baseline_class0_recall = 0.7303  # From error analysis
class0_improvement = class0_recall - baseline_class0_recall

print(f"\nClass 0 Recall:")
print(f"   Baseline (394 features): {baseline_class0_recall:.4f} ({baseline_class0_recall*100:.2f}%)")
print(f"   Current (522 features):  {class0_recall:.4f} ({class0_recall*100:.2f}%)")
print(f"   Improvement:             {class0_improvement:+.4f} ({class0_improvement*100:+.2f} pts)")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
results = {
    'test_f1': float(test_f1),
    'test_acc': float(test_acc),
    'val_f1': float(best_val_f1),
    'class0_recall': float(class0_recall),
    'class1_recall': float(class1_recall),
    'confusion_matrix': cm.tolist(),
    'improvement_vs_single': float(improvement_vs_single),
    'improvement_vs_ensemble': float(improvement_vs_ensemble),
    'class0_improvement': float(class0_improvement),
    'features': {
        'total': 522,
        'original': 384,
        'graph_stats': 10,
        'node2vec': 128
    },
    'hyperparameters': {
        'hidden_channels': 256,
        'num_heads': 10,
        'num_layers': 3,
        'dropout': 0.25,
        'lr': 0.0016,
        'weight_decay': 0.000134,
        'seed': 42
    }
}

output_file = output_dir / 'results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {output_file}")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

target_f1 = 0.92
achieved_target = test_f1 >= target_f1

print(f"\n🎯 Target: {target_f1:.4f} (92.00%)")
print(f"   Achieved: {test_f1:.4f} ({test_f1*100:.2f}%)")

if achieved_target:
    print(f"\n🎉 SUCCESS! Achieved target F1 of {target_f1:.2%}!")
    print(f"   Improvement: {improvement_vs_ensemble*100:+.2f} pts over ensemble baseline")
elif test_f1 >= baseline_ensemble:
    print(f"\n✅ PROGRESS! Improved over ensemble baseline!")
    print(f"   Gain: {improvement_vs_ensemble*100:+.2f} pts")
    print(f"   Gap to 92% target: {(target_f1 - test_f1)*100:.2f} pts")
else:
    print(f"\n⚠️  No improvement over ensemble baseline")
    print(f"   Current: {test_f1:.4f}")
    print(f"   Baseline: {baseline_ensemble:.4f}")
    print(f"   Difference: {improvement_vs_ensemble*100:.2f} pts")

print(f"\n📊 Key Metrics:")
print(f"   Overall F1:      {test_f1:.4f}")
print(f"   Class 0 recall:  {class0_recall:.4f} (was {baseline_class0_recall:.4f})")
print(f"   Class 1 recall:  {class1_recall:.4f}")

if class0_improvement > 0:
    print(f"\n✅ Class 0 improved by {class0_improvement*100:.2f} pts!")
else:
    print(f"\n⚠️  Class 0 did not improve")

print("\n" + "=" * 80)
