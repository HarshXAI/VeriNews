"""
Verify baseline model and extract the exact data splits it used
"""

import torch
import json
from pathlib import Path

# Load the baseline model checkpoint
checkpoint_path = "experiments/models_fullscale/gat_model_best_scaled.pt"
data_path = "data/graphs_full/graph_data_enriched.pt"

print("="*80)
print("VERIFYING BASELINE MODEL AND SPLITS")
print("="*80)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n📦 Checkpoint contents:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Load data
data = torch.load(data_path, map_location='cpu', weights_only=False)

print(f"\n📊 Data info:")
print(f"  Nodes: {data.x.shape[0]}")
print(f"  Edges: {data.edge_index.shape[1]}")
print(f"  Features: {data.x.shape[1]}")

# Check if data has masks
if hasattr(data, 'train_mask'):
    print(f"\n✅ Data HAS masks:")
    print(f"  Train: {data.train_mask.sum()}")
    print(f"  Val: {data.val_mask.sum()}")
    print(f"  Test: {data.test_mask.sum()}")
    
    # Save these masks
    masks = {
        'train_mask': data.train_mask,
        'val_mask': data.val_mask,
        'test_mask': data.test_mask
    }
    
    torch.save(masks, 'experiments/models_fullscale/baseline_splits.pt')
    print(f"\n💾 Saved baseline splits to experiments/models_fullscale/baseline_splits.pt")
else:
    print(f"\n❌ Data does NOT have masks - need to regenerate")

# Load metrics
with open('experiments/models_fullscale/training_metrics_scaled.json', 'r') as f:
    metrics = json.load(f)

print(f"\n📊 Baseline metrics:")
print(f"  Best epoch: {metrics['best_epoch']}")
print(f"  Test F1: {metrics['test_f1']:.4f} ({metrics['test_f1']*100:.2f}%)")
print(f"  Test accuracy: {metrics['test_accuracy']:.4f}")
print(f"  Test precision: {metrics['test_precision']:.4f}")
print(f"  Test recall: {metrics['test_recall']:.4f}")

print("\n" + "="*80)
