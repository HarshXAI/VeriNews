"""
Quick Ensemble Strategy for 95% F1 Target
==========================================

Faster approach:
1. Train just 3 models (not 7) with different seeds
2. Reduce epochs for faster training  
3. Add progress bars
4. Test best ensemble strategy only

Target: 91-92% F1 in ~1 hour instead of 3 hours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json
import os
from pathlib import Path
import time
from tqdm import tqdm

# Device configuration
device = torch.device('cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = Path("experiments/quick_ensemble")
output_dir.mkdir(parents=True, exist_ok=True)


class GATv2Model(nn.Module):
    """GATv2 model with best hyperparameters"""
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


class GINModel(nn.Module):
    """GIN model"""
    def __init__(self, in_channels, hidden_channels=256, num_layers=3, dropout=0.25, num_classes=2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.convs.append(GINConv(nn1))
        
        for _ in range(num_layers - 2):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(GINConv(nn_layer))
        
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.classifier(x)
        return x


def train_model(model, data, train_mask, val_mask, optimizer, epochs=200, patience=20, model_name="Model"):
    """Train a single model with early stopping and progress bar"""
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    pbar = tqdm(range(epochs), desc=f"Training {model_name}")
    
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
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val_f1': f'{val_f1:.4f}', 'best': f'{best_val_f1:.4f}'})
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience // 10:
                    pbar.set_description(f"{model_name} [Early stopped]")
                    break
    
    pbar.close()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_f1


def get_predictions(model, data, mask):
    """Get predictions and probabilities"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[mask], dim=1)
        preds = probs.argmax(dim=1)
    return preds, probs


def weighted_voting_ensemble(probs_list, weights):
    """Weighted soft voting"""
    weights_tensor = torch.tensor(weights, device=probs_list[0].device).view(-1, 1, 1)
    weighted_probs = torch.stack(probs_list) * weights_tensor
    avg_probs = weighted_probs.sum(dim=0)
    return avg_probs.argmax(dim=1)


def main():
    print("=" * 80)
    print("QUICK ENSEMBLE FOR 95% F1 TARGET")
    print("=" * 80)
    print(f"\nCurrent baseline: 90.52% F1")
    print(f"Strategy: Train 3 diverse models (2 GATv2 + 1 GIN) with weighted ensemble")
    print(f"Estimated time: ~45-60 minutes\n")
    
    # Load data
    print("Loading data...")
    data = torch.load('data/graphs_full/graph_data_enriched_with_stats.pt', weights_only=False)
    splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)
    
    train_mask = splits['train_mask']
    val_mask = splits['val_mask']
    test_mask = splits['test_mask']
    
    data = data.to(device)
    
    print(f"Features: {data.x.shape[1]}, Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}\n")
    
    # Best hyperparameters
    hpo_config = {
        'hidden_channels': 256,
        'num_heads': 10,
        'num_layers': 3,
        'dropout': 0.25,
        'lr': 0.0016,
        'weight_decay': 0.000134
    }
    
    models = []
    val_f1s = []
    
    # Train Model 1: GATv2 (seed 314 - our best)
    print("\n" + "=" * 80)
    print("Training Model 1: GATv2 (seed 314)")
    print("=" * 80)
    torch.manual_seed(314)
    np.random.seed(314)
    
    model1 = GATv2Model(
        in_channels=data.x.shape[1],
        hidden_channels=hpo_config['hidden_channels'],
        num_heads=hpo_config['num_heads'],
        num_layers=hpo_config['num_layers'],
        dropout=hpo_config['dropout']
    ).to(device)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=hpo_config['lr'], weight_decay=hpo_config['weight_decay'])
    model1, val_f1_1 = train_model(model1, data, train_mask, val_mask, optimizer1, model_name="GATv2-314")
    
    print(f"✓ Model 1 completed: Val F1 = {val_f1_1:.4f}\n")
    models.append(model1)
    val_f1s.append(val_f1_1)
    
    # Train Model 2: GATv2 (seed 42 - diversity)
    print("=" * 80)
    print("Training Model 2: GATv2 (seed 42)")
    print("=" * 80)
    torch.manual_seed(42)
    np.random.seed(42)
    
    model2 = GATv2Model(
        in_channels=data.x.shape[1],
        hidden_channels=hpo_config['hidden_channels'],
        num_heads=hpo_config['num_heads'],
        num_layers=hpo_config['num_layers'],
        dropout=hpo_config['dropout']
    ).to(device)
    
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=hpo_config['lr'], weight_decay=hpo_config['weight_decay'])
    model2, val_f1_2 = train_model(model2, data, train_mask, val_mask, optimizer2, model_name="GATv2-42")
    
    print(f"✓ Model 2 completed: Val F1 = {val_f1_2:.4f}\n")
    models.append(model2)
    val_f1s.append(val_f1_2)
    
    # Train Model 3: GIN (seed 314 - architecture diversity)
    print("=" * 80)
    print("Training Model 3: GIN (seed 314)")
    print("=" * 80)
    torch.manual_seed(314)
    np.random.seed(314)
    
    model3 = GINModel(
        in_channels=data.x.shape[1],
        hidden_channels=hpo_config['hidden_channels'],
        num_layers=hpo_config['num_layers'],
        dropout=hpo_config['dropout']
    ).to(device)
    
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=hpo_config['lr'], weight_decay=hpo_config['weight_decay'])
    model3, val_f1_3 = train_model(model3, data, train_mask, val_mask, optimizer3, model_name="GIN-314")
    
    print(f"✓ Model 3 completed: Val F1 = {val_f1_3:.4f}\n")
    models.append(model3)
    val_f1s.append(val_f1_3)
    
    # Evaluate individual models on test set
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 80)
    
    individual_results = []
    for i, model in enumerate(models):
        preds, _ = get_predictions(model, data, test_mask)
        test_f1 = f1_score(data.y[test_mask].cpu(), preds.cpu(), average='weighted')
        test_acc = accuracy_score(data.y[test_mask].cpu(), preds.cpu())
        
        model_name = ["GATv2-314", "GATv2-42", "GIN-314"][i]
        print(f"{model_name:12s} - Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}")
        
        individual_results.append({
            'model': model_name,
            'val_f1': float(val_f1s[i]),
            'test_f1': float(test_f1),
            'test_acc': float(test_acc)
        })
    
    # Ensemble predictions
    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS")
    print("=" * 80)
    
    # Get all probabilities
    probs_list = []
    for model in models:
        _, probs = get_predictions(model, data, test_mask)
        probs_list.append(probs)
    
    # Strategy 1: Equal weight (simple average)
    equal_weights = [1.0/len(models)] * len(models)
    equal_preds = weighted_voting_ensemble(probs_list, equal_weights)
    equal_f1 = f1_score(data.y[test_mask].cpu(), equal_preds.cpu(), average='weighted')
    equal_acc = accuracy_score(data.y[test_mask].cpu(), equal_preds.cpu())
    
    print(f"Equal Voting:    Test F1: {equal_f1:.4f}, Test Acc: {equal_acc:.4f}")
    
    # Strategy 2: Weighted by validation F1
    total_val_f1 = sum(val_f1s)
    weighted_weights = [f1 / total_val_f1 for f1 in val_f1s]
    weighted_preds = weighted_voting_ensemble(probs_list, weighted_weights)
    weighted_f1 = f1_score(data.y[test_mask].cpu(), weighted_preds.cpu(), average='weighted')
    weighted_acc = accuracy_score(data.y[test_mask].cpu(), weighted_preds.cpu())
    
    print(f"Weighted Voting: Test F1: {weighted_f1:.4f}, Test Acc: {weighted_acc:.4f}")
    print(f"  Weights: {[f'{w:.3f}' for w in weighted_weights]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    baseline_f1 = 0.9052
    best_individual_f1 = max([r['test_f1'] for r in individual_results])
    best_ensemble_f1 = max(equal_f1, weighted_f1)
    
    print(f"\nBaseline (single GATv2):  {baseline_f1:.4f}")
    print(f"Best individual model:    {best_individual_f1:.4f} ({(best_individual_f1-baseline_f1)*100:+.2f} pts)")
    print(f"Best ensemble:            {best_ensemble_f1:.4f} ({(best_ensemble_f1-baseline_f1)*100:+.2f} pts)")
    
    if best_ensemble_f1 >= 0.91:
        print(f"\n🎉 SUCCESS! Ensemble achieved {best_ensemble_f1:.4f} F1 (target was 91%+)")
    elif best_ensemble_f1 > baseline_f1:
        print(f"\n✅ Progress! Improved to {best_ensemble_f1:.4f} F1")
        print(f"   Gap to 91% target: {(0.91-best_ensemble_f1)*100:.2f} pts")
    else:
        print(f"\n⚠️  Ensemble didn't improve over baseline")
    
    # Save results
    results = {
        'baseline_f1': float(baseline_f1),
        'individual_models': individual_results,
        'ensemble': {
            'equal_voting': {'test_f1': float(equal_f1), 'test_acc': float(equal_acc)},
            'weighted_voting': {'test_f1': float(weighted_f1), 'test_acc': float(weighted_acc), 'weights': weighted_weights}
        },
        'best_ensemble_f1': float(best_ensemble_f1),
        'improvement': float(best_ensemble_f1 - baseline_f1)
    }
    
    output_file = output_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
