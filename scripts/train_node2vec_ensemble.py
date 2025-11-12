"""
Train ensemble with Node2Vec embeddings (522 features)
Expected: 91.5-92.0% F1 by combining Class 0 improvement with ensemble diversity
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("=" * 80)
print("NODE2VEC ENSEMBLE TRAINING")
print("=" * 80)
print()

# Load data with Node2Vec embeddings
print("1. Loading data with Node2Vec embeddings...")
data = torch.load('data/graphs_full/graph_data_with_node2vec.pt', weights_only=False)
splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)

train_mask = splits['train_mask']
val_mask = splits['val_mask']
test_mask = splits['test_mask']

print(f"Features: {data.x.shape[1]} (394 baseline + 128 Node2Vec)")
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
print()

# Model definitions
class GATv2Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_heads=10, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, 
                                       heads=num_heads, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Output layer
        self.convs.append(GATv2Conv(hidden_channels * num_heads, 2, heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn1))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.lin = torch.nn.Linear(hidden_channels, 2)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[train_mask], data.y[train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    return f1, acc, out[mask].cpu()

def train_model(model_name, model, data, train_mask, val_mask, seed, epochs=200, patience=20):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_name} (seed {seed})")
    print(f"{'='*80}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    pbar = tqdm(range(epochs), desc=f'{model_name}')
    for epoch in pbar:
        loss = train_epoch(model, data, optimizer, train_mask)
        val_f1, val_acc, _ = evaluate(model, data, val_mask)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'val_f1': f'{val_f1:.4f}',
            'best': f'{best_val_f1:.4f}'
        })
        
        if patience_counter >= patience:
            print(f"\n✅ Early stopped at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"✅ Best validation F1: {best_val_f1:.4f}")
    
    return model, best_val_f1

# Train 3 models with different seeds
print("=" * 80)
print("TRAINING INDIVIDUAL MODELS")
print("=" * 80)

models_info = [
    ('GATv2-42', GATv2Model, 42),
    ('GATv2-314', GATv2Model, 314),
    ('GIN-999', GINModel, 999),
]

trained_models = []
val_f1s = []

for model_name, ModelClass, seed in models_info:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ModelClass(in_channels=data.x.shape[1]).to(device)
    data = data.to(device)
    
    model, val_f1 = train_model(model_name, model, data, train_mask, val_mask, seed)
    
    trained_models.append((model_name, model, seed))
    val_f1s.append(val_f1)

# Individual model evaluation on test set
print("\n" + "=" * 80)
print("INDIVIDUAL MODEL PERFORMANCE")
print("=" * 80)
print()

individual_results = []
all_test_probs = []

for (model_name, model, seed), val_f1 in zip(trained_models, val_f1s):
    test_f1, test_acc, test_probs = evaluate(model, data, test_mask)
    all_test_probs.append(test_probs)
    
    pred = test_probs.argmax(dim=1).numpy()
    y_true = data.y[test_mask].cpu().numpy()
    
    cm = confusion_matrix(y_true, pred)
    class_0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    
    individual_results.append({
        'model': model_name,
        'seed': seed,
        'val_f1': float(val_f1),
        'test_f1': float(test_f1),
        'test_acc': float(test_acc),
        'class_0_recall': float(class_0_recall),
        'class_1_recall': float(class_1_recall)
    })
    
    print(f"{model_name} (seed {seed}):")
    print(f"  Val F1:  {val_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  Class 0 recall: {class_0_recall:.4f}")
    print(f"  Class 1 recall: {class_1_recall:.4f}")
    print()

# Ensemble predictions
print("=" * 80)
print("ENSEMBLE EVALUATION")
print("=" * 80)
print()

# Average probabilities
ensemble_probs = torch.stack(all_test_probs).mean(dim=0)
ensemble_pred = ensemble_probs.argmax(dim=1).numpy()
y_true = data.y[test_mask].cpu().numpy()

# Calculate metrics
ensemble_f1 = f1_score(y_true, ensemble_pred, average='macro')
ensemble_acc = accuracy_score(y_true, ensemble_pred)

# Confusion matrix
cm = confusion_matrix(y_true, ensemble_pred)
class_0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

print(f"📊 Ensemble Performance:")
print(f"   Test F1:       {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
print(f"   Test Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
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

# Comparison with baselines
print("=" * 80)
print("COMPARISON WITH BASELINES")
print("=" * 80)
print()

baseline_394_single = 0.9052
baseline_394_ensemble = 0.9126
baseline_522_single = 0.9113

print(f"Baseline (394 features, single model): {baseline_394_single:.4f}")
print(f"Baseline (394 features, ensemble):     {baseline_394_ensemble:.4f}")
print(f"Current (522 features, single best):   {max([r['test_f1'] for r in individual_results]):.4f}")
print(f"Current (522 features, ensemble):      {ensemble_f1:.4f}")
print()
print(f"Improvement vs 394 ensemble: {ensemble_f1 - baseline_394_ensemble:+.4f} ({(ensemble_f1 - baseline_394_ensemble)*100:+.2f} pts)")
print(f"Improvement vs 522 single:   {ensemble_f1 - baseline_522_single:+.4f} ({(ensemble_f1 - baseline_522_single)*100:+.2f} pts)")
print()

# Class 0 improvement
baseline_class0 = 0.7303
print(f"Class 0 Recall:")
print(f"   Baseline (394 features): {baseline_class0:.4f} ({baseline_class0*100:.2f}%)")
print(f"   Current (522 features):  {class_0_recall:.4f} ({class_0_recall*100:.2f}%)")
print(f"   Improvement:             {class_0_recall - baseline_class0:+.4f} ({(class_0_recall - baseline_class0)*100:+.2f} pts)")
print()

# Save results
output_dir = Path('experiments/node2vec_ensemble')
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'ensemble': {
        'test_f1': float(ensemble_f1),
        'test_acc': float(ensemble_acc),
        'class_0_recall': float(class_0_recall),
        'class_1_recall': float(class_1_recall),
        'confusion_matrix': cm.tolist()
    },
    'individual_models': individual_results,
    'comparison': {
        'baseline_394_single': float(baseline_394_single),
        'baseline_394_ensemble': float(baseline_394_ensemble),
        'baseline_522_single': float(baseline_522_single),
        'current_522_ensemble': float(ensemble_f1),
        'improvement_vs_394_ensemble': float(ensemble_f1 - baseline_394_ensemble),
        'improvement_vs_522_single': float(ensemble_f1 - baseline_522_single),
        'class_0_improvement': float(class_0_recall - baseline_class0)
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
print(f"🎯 Target: 0.9200 (92.00%)")
print(f"   Achieved: {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
print()

if ensemble_f1 >= 0.92:
    print("🎉 SUCCESS! Target reached!")
elif ensemble_f1 >= 0.915:
    print("✅ Very close! Within 0.5 pts of target")
elif ensemble_f1 > baseline_394_ensemble:
    print("✅ Improvement over baseline!")
else:
    print("⚠️  No improvement over baseline")

print()
print(f"📊 Key Metrics:")
print(f"   Overall F1:      {ensemble_f1:.4f}")
print(f"   Class 0 recall:  {class_0_recall:.4f} (was {baseline_class0:.4f})")
print(f"   Class 1 recall:  {class_1_recall:.4f}")
print()
print("=" * 80)
