# Explainability and Visualization

This notebook demonstrates how to interpret and visualize GAT model predictions.

## 1. Setup

```python
import sys
sys.path.append('..')

import torch
import networkx as nx

from src.models import FakeNewsGAT
from src.evaluation import AttentionAnalyzer
from src.visualization import GraphVisualizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 2. Load Trained Model

```python
# Load model
model = FakeNewsGAT(
    in_channels=768,
    hidden_channels=128,
    out_channels=2,
    num_layers=3,
    num_heads=8
).to(device)

# Load checkpoint
checkpoint = torch.load('../experiments/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 3. Analyze Attention Weights

```python
# Load test data
data = torch.load('../data/graphs/test_sample.pt')

# Get attention weights
analyzer = AttentionAnalyzer(model, device)
attention_weights = analyzer.get_attention_weights(
    data.x,
    data.edge_index,
    data.batch
)

print(f"Number of attention layers: {len(attention_weights)}")
```

## 4. Identify Key Spreaders

```python
# Find influential users
influential_users = analyzer.identify_influential_users(
    attention_weights,
    top_k=20
)

print("Top 20 Key Spreaders:")
for i, (user_id, score) in enumerate(influential_users, 1):
    print(f"{i:2d}. User {user_id:6d}: {score:.4f}")
```

## 5. Visualize Propagation Tree

```python
# Build propagation tree
edge_idx, attn = attention_weights[0]
tree = analyzer.build_propagation_tree(
    edge_idx,
    attn,
    root_node=0
)

# Visualize
visualizer = GraphVisualizer(figsize=(14, 10))
visualizer.plot_propagation_tree(
    tree,
    root_node=0,
    save_path='../outputs/propagation_tree.png'
)
```

## 6. Compute Propagation Metrics

```python
# Compute metrics
metrics = analyzer.compute_propagation_metrics(tree)

print("Propagation Metrics:")
print(f"  Depth: {metrics['depth']}")
print(f"  Breadth: {metrics['breadth']}")
print(f"  Viral Coefficient: {metrics['viral_coefficient']:.2f}")
```

## 7. Visualize Attention Heatmap

```python
import numpy as np

# Convert attention to matrix
edge_idx, attn = attention_weights[0]
edge_idx = edge_idx.cpu().numpy()
attn = attn.cpu().numpy()

# Create attention matrix
num_nodes = data.x.size(0)
attention_matrix = np.zeros((num_nodes, num_nodes))

for i in range(edge_idx.shape[1]):
    src, tgt = edge_idx[0, i], edge_idx[1, i]
    attention_matrix[src, tgt] = attn[i].mean()

# Plot heatmap (for a subset of nodes)
visualizer.plot_attention_heatmap(
    attention_matrix[:50, :50],
    save_path='../outputs/attention_heatmap.png'
)
```

## 8. Feature Importance

```python
from src.evaluation import FeatureImportanceAnalyzer

# Compute feature importance
feature_names = ['text_emb_' + str(i) for i in range(768)]
importance = FeatureImportanceAnalyzer.compute_feature_importance(
    model,
    test_loader,
    feature_names,
    device
)

# Show top features
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("Top 10 Important Features:")
for name, score in sorted_features[:10]:
    print(f"  {name}: {score:.4f}")
```

## 9. Generate Report

```python
# Create comprehensive report
report = {
    'model': 'FakeNewsGAT',
    'test_accuracy': 0.XX,
    'test_f1': 0.XX,
    'top_spreaders': influential_users[:10],
    'propagation_metrics': metrics
}

import json
with open('../outputs/explainability_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Report saved to outputs/explainability_report.json")
```
