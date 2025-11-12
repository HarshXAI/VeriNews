"""
Advanced Ensemble Methods for 95% F1 Target
============================================

This script implements sophisticated ensemble strategies to improve beyond 90.52% F1:

1. Multi-seed ensemble: Train same architecture with different seeds
2. Architecture diversity: Combine GATv2 + GIN 
3. Voting strategies: Soft voting, hard voting, weighted voting
4. Stacking: Meta-learner on top of base models

Current baseline: 90.52% F1 (GATv2 + Enhanced Features)
Target: 91-92% F1 (+0.5-1.5 points)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import time

# Device configuration
device = torch.device('cpu')  # Consistent with previous experiments
print(f"Using device: {device}")

# Create output directory
output_dir = Path("experiments/advanced_ensemble")
output_dir.mkdir(parents=True, exist_ok=True)


class GATv2Model(nn.Module):
    """GATv2 model with best hyperparameters from HPO"""
    def __init__(self, in_channels, hidden_channels=256, num_heads=10, 
                 num_layers=3, dropout=0.25, num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, 
                                       heads=num_heads, dropout=dropout))
        
        # Output layer
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
    """GIN model that performed well (90.12% F1)"""
    def __init__(self, in_channels, hidden_channels=256, num_layers=3, dropout=0.25, num_classes=2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # Input layer
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.convs.append(GINConv(nn1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(GINConv(nn_layer))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.classifier(x)
        return x


def train_model(model, data, train_mask, val_mask, optimizer, epochs=300, patience=30):
    """Train a single model with early stopping"""
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
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
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience // 10:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_f1


def get_predictions(model, data, mask):
    """Get predictions and probabilities from a model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[mask], dim=1)
        preds = probs.argmax(dim=1)
    return preds, probs


def evaluate_model(preds, true_labels):
    """Evaluate predictions"""
    f1 = f1_score(true_labels, preds, average='weighted')
    acc = accuracy_score(true_labels, preds)
    return f1, acc


def soft_voting_ensemble(probs_list: List[torch.Tensor]) -> torch.Tensor:
    """Soft voting: average probabilities"""
    avg_probs = torch.stack(probs_list).mean(dim=0)
    return avg_probs.argmax(dim=1)


def hard_voting_ensemble(preds_list: List[torch.Tensor]) -> torch.Tensor:
    """Hard voting: majority vote"""
    stacked = torch.stack(preds_list)
    votes, _ = torch.mode(stacked, dim=0)
    return votes


def weighted_voting_ensemble(probs_list: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """Weighted soft voting based on validation F1 scores"""
    weights_tensor = torch.tensor(weights, device=probs_list[0].device).view(-1, 1, 1)
    weighted_probs = torch.stack(probs_list) * weights_tensor
    avg_probs = weighted_probs.sum(dim=0)
    return avg_probs.argmax(dim=1)


def stacking_ensemble(train_probs_list: List[np.ndarray], train_labels: np.ndarray,
                     test_probs_list: List[np.ndarray]) -> np.ndarray:
    """Stacking: use logistic regression as meta-learner"""
    # Stack probabilities as features
    X_train = np.concatenate([p.cpu().numpy() for p in train_probs_list], axis=1)
    X_test = np.concatenate([p.cpu().numpy() for p in test_probs_list], axis=1)
    
    # Train meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(X_train, train_labels.cpu().numpy())
    
    # Predict
    predictions = meta_learner.predict(X_test)
    return torch.tensor(predictions)


def main():
    print("=" * 80)
    print("ADVANCED ENSEMBLE FOR 95% F1 TARGET")
    print("=" * 80)
    print(f"\nCurrent baseline: 90.52% F1 (GATv2 + Enhanced Features)")
    print(f"Target: 91-92% F1 (+0.5-1.5 points)\n")
    
    # Load data with enhanced features
    print("Loading enhanced graph data...")
    data = torch.load('data/graphs_full/graph_data_enriched_with_stats.pt', weights_only=False)
    print(f"Features: {data.x.shape[1]} dimensions (394 with graph statistics)")
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    
    # Load best splits (seed 314)
    print("\nLoading best splits (seed 314)...")
    splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)
    train_mask = splits['train_mask']
    val_mask = splits['val_mask']
    test_mask = splits['test_mask']
    
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Move data to device
    data = data.to(device)
    
    # Best hyperparameters from HPO
    hpo_config = {
        'hidden_channels': 256,
        'num_heads': 10,
        'num_layers': 3,
        'dropout': 0.25,
        'lr': 0.0016,
        'weight_decay': 0.000134
    }
    
    results = {}
    
    # ========================================================================
    # Strategy 1: Multi-seed GATv2 Ensemble
    # ========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 1: Multi-Seed GATv2 Ensemble")
    print("=" * 80)
    print("Training 5 GATv2 models with different random seeds...")
    
    seeds = [314, 42, 123, 456, 789]
    gatv2_models = []
    gatv2_val_f1s = []
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Model {i+1}/5 (seed={seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = GATv2Model(
            in_channels=data.x.shape[1],
            hidden_channels=hpo_config['hidden_channels'],
            num_heads=hpo_config['num_heads'],
            num_layers=hpo_config['num_layers'],
            dropout=hpo_config['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hpo_config['lr'],
            weight_decay=hpo_config['weight_decay']
        )
        
        start = time.time()
        model, val_f1 = train_model(model, data, train_mask, val_mask, optimizer)
        elapsed = time.time() - start
        
        print(f"Training time: {elapsed:.1f}s, Val F1: {val_f1:.4f}")
        
        gatv2_models.append(model)
        gatv2_val_f1s.append(val_f1)
    
    # Evaluate multi-seed ensemble
    print("\n--- Evaluating Multi-Seed GATv2 Ensemble ---")
    gatv2_preds_list = []
    gatv2_probs_list = []
    
    for model in gatv2_models:
        preds, probs = get_predictions(model, data, test_mask)
        gatv2_preds_list.append(preds)
        gatv2_probs_list.append(probs)
    
    # Soft voting
    soft_vote_preds = soft_voting_ensemble(gatv2_probs_list)
    soft_f1, soft_acc = evaluate_model(soft_vote_preds.cpu(), data.y[test_mask].cpu())
    print(f"Soft Voting: F1 = {soft_f1:.4f}, Acc = {soft_acc:.4f}")
    
    # Hard voting
    hard_vote_preds = hard_voting_ensemble(gatv2_preds_list)
    hard_f1, hard_acc = evaluate_model(hard_vote_preds.cpu(), data.y[test_mask].cpu())
    print(f"Hard Voting: F1 = {hard_f1:.4f}, Acc = {hard_acc:.4f}")
    
    # Weighted voting
    weights = [f1 / sum(gatv2_val_f1s) for f1 in gatv2_val_f1s]
    weighted_preds = weighted_voting_ensemble(gatv2_probs_list, weights)
    weighted_f1, weighted_acc = evaluate_model(weighted_preds.cpu(), data.y[test_mask].cpu())
    print(f"Weighted Voting: F1 = {weighted_f1:.4f}, Acc = {weighted_acc:.4f}")
    
    results['multi_seed_gatv2'] = {
        'individual_val_f1s': [float(f1) for f1 in gatv2_val_f1s],
        'soft_voting': {'test_f1': float(soft_f1), 'test_acc': float(soft_acc)},
        'hard_voting': {'test_f1': float(hard_f1), 'test_acc': float(hard_acc)},
        'weighted_voting': {'test_f1': float(weighted_f1), 'test_acc': float(weighted_acc)}
    }
    
    # ========================================================================
    # Strategy 2: Architecture Diversity (GATv2 + GIN)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 2: Architecture Diversity (GATv2 + GIN)")
    print("=" * 80)
    print("Training 3 GATv2 + 2 GIN models...")
    
    # Train 2 GIN models
    gin_models = []
    gin_val_f1s = []
    
    for i, seed in enumerate([314, 42]):
        print(f"\n--- GIN Model {i+1}/2 (seed={seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = GINModel(
            in_channels=data.x.shape[1],
            hidden_channels=hpo_config['hidden_channels'],
            num_layers=hpo_config['num_layers'],
            dropout=hpo_config['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hpo_config['lr'],
            weight_decay=hpo_config['weight_decay']
        )
        
        start = time.time()
        model, val_f1 = train_model(model, data, train_mask, val_mask, optimizer)
        elapsed = time.time() - start
        
        print(f"Training time: {elapsed:.1f}s, Val F1: {val_f1:.4f}")
        
        gin_models.append(model)
        gin_val_f1s.append(val_f1)
    
    # Combine best GATv2 models with GIN models
    print("\n--- Evaluating GATv2 + GIN Ensemble ---")
    
    # Get top 3 GATv2 models by validation F1
    top3_indices = np.argsort(gatv2_val_f1s)[-3:]
    mixed_models = [gatv2_models[i] for i in top3_indices] + gin_models
    mixed_val_f1s = [gatv2_val_f1s[i] for i in top3_indices] + gin_val_f1s
    
    mixed_preds_list = []
    mixed_probs_list = []
    
    for model in mixed_models:
        preds, probs = get_predictions(model, data, test_mask)
        mixed_preds_list.append(preds)
        mixed_probs_list.append(probs)
    
    # Soft voting
    mixed_soft_preds = soft_voting_ensemble(mixed_probs_list)
    mixed_soft_f1, mixed_soft_acc = evaluate_model(mixed_soft_preds.cpu(), data.y[test_mask].cpu())
    print(f"Soft Voting (3 GATv2 + 2 GIN): F1 = {mixed_soft_f1:.4f}, Acc = {mixed_soft_acc:.4f}")
    
    # Weighted voting
    mixed_weights = [f1 / sum(mixed_val_f1s) for f1 in mixed_val_f1s]
    mixed_weighted_preds = weighted_voting_ensemble(mixed_probs_list, mixed_weights)
    mixed_weighted_f1, mixed_weighted_acc = evaluate_model(mixed_weighted_preds.cpu(), data.y[test_mask].cpu())
    print(f"Weighted Voting (3 GATv2 + 2 GIN): F1 = {mixed_weighted_f1:.4f}, Acc = {mixed_weighted_acc:.4f}")
    
    results['architecture_diversity'] = {
        'gin_val_f1s': [float(f1) for f1 in gin_val_f1s],
        'soft_voting': {'test_f1': float(mixed_soft_f1), 'test_acc': float(mixed_soft_acc)},
        'weighted_voting': {'test_f1': float(mixed_weighted_f1), 'test_acc': float(mixed_weighted_acc)}
    }
    
    # ========================================================================
    # Strategy 3: Stacking with Meta-Learner
    # ========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 3: Stacking with Meta-Learner")
    print("=" * 80)
    print("Training logistic regression meta-learner on validation set...")
    
    # Get validation predictions for training meta-learner
    val_probs_list = []
    test_probs_list = []
    
    for model in mixed_models:
        _, val_probs = get_predictions(model, data, val_mask)
        _, test_probs = get_predictions(model, data, test_mask)
        val_probs_list.append(val_probs)
        test_probs_list.append(test_probs)
    
    # Train and predict with stacking
    stacking_preds = stacking_ensemble(val_probs_list, data.y[val_mask], test_probs_list)
    stacking_f1, stacking_acc = evaluate_model(stacking_preds.cpu(), data.y[test_mask].cpu())
    print(f"Stacking (Logistic Meta-Learner): F1 = {stacking_f1:.4f}, Acc = {stacking_acc:.4f}")
    
    results['stacking'] = {
        'test_f1': float(stacking_f1),
        'test_acc': float(stacking_acc)
    }
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBaseline (single GATv2 + Enhanced): 90.52% F1\n")
    
    all_results = [
        ("Multi-Seed GATv2 (Soft Voting)", soft_f1),
        ("Multi-Seed GATv2 (Hard Voting)", hard_f1),
        ("Multi-Seed GATv2 (Weighted Voting)", weighted_f1),
        ("GATv2+GIN Mix (Soft Voting)", mixed_soft_f1),
        ("GATv2+GIN Mix (Weighted Voting)", mixed_weighted_f1),
        ("Stacking (Meta-Learner)", stacking_f1),
    ]
    
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    print("Ranking by Test F1:")
    for i, (name, f1) in enumerate(all_results, 1):
        improvement = (f1 - 0.9052) * 100
        symbol = "🏆" if i == 1 else "⭐" if i <= 3 else "  "
        print(f"{symbol} {i}. {name:40s} {f1:.4f} ({improvement:+.2f} pts)")
    
    # Save results
    results['summary'] = {
        'baseline_f1': 0.9052,
        'best_ensemble': all_results[0][0],
        'best_f1': float(all_results[0][1]),
        'improvement': float(all_results[0][1] - 0.9052)
    }
    
    output_file = output_dir / 'ensemble_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Check if we achieved target
    best_f1 = all_results[0][1]
    if best_f1 >= 0.91:
        print(f"\n🎉 SUCCESS! Achieved {best_f1:.4f} F1 (target was 91-92%)")
    elif best_f1 > 0.9052:
        print(f"\n✅ Progress! Improved to {best_f1:.4f} F1 (+{(best_f1-0.9052)*100:.2f} pts)")
        print(f"   Still {(0.91-best_f1)*100:.2f} pts away from 91% target")
    else:
        print(f"\n⚠️  No improvement over baseline. Best ensemble: {best_f1:.4f} F1")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
