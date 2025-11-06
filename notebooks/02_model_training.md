# Model Training and Evaluation

This notebook demonstrates training a GAT model for fake news detection.

## 1. Setup

```python
import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.models import FakeNewsGAT
from src.training import GATTrainer
from src.evaluation import ModelEvaluator
from src.visualization import MetricsVisualizer

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
```

## 2. Load Preprocessed Data

```python
# Load graph data
data = torch.load('../data/graphs/processed_graphs.pt')

# Split into train/val/test
from torch_geometric.transforms import RandomNodeSplit

transform = RandomNodeSplit(split='train_rest', num_val=0.15, num_test=0.15)
data = transform(data)

print(f"Training nodes: {data.train_mask.sum().item()}")
print(f"Validation nodes: {data.val_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")
```

## 3. Create Model

```python
model = FakeNewsGAT(
    in_channels=768,  # BERT embedding dimension
    hidden_channels=128,
    out_channels=2,
    num_layers=3,
    num_heads=8,
    dropout=0.3
).to(device)

print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 4. Train Model

```python
# Create data loaders
train_loader = DataLoader([data], batch_size=1)
val_loader = DataLoader([data], batch_size=1)

# Create trainer
trainer = GATTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=0.001,
    device=device
)

# Train
trainer.train(epochs=100, early_stopping_patience=10)
```

## 5. Evaluate Model

```python
# Load best model
trainer.load_checkpoint('best_model.pt')

# Evaluate
evaluator = ModelEvaluator(model, device)
metrics = evaluator.evaluate_and_report(val_loader)
```

## 6. Visualize Results

```python
# Plot confusion matrix
visualizer = MetricsVisualizer()
visualizer.plot_confusion_matrix(
    metrics['confusion_matrix'],
    class_names=['Fake', 'Real']
)

# Plot metrics
visualizer.plot_metrics_comparison(metrics)
```

## 7. Attention Analysis

```python
from src.evaluation import AttentionAnalyzer

analyzer = AttentionAnalyzer(model, device)
attention_weights = analyzer.get_attention_weights(data.x, data.edge_index)

# Identify influential users
influential = analyzer.identify_influential_users(attention_weights, top_k=20)
print("Top 20 influential users:")
for user_id, score in influential:
    print(f"  User {user_id}: {score:.4f}")
```
