"""
Hyperparameter Optimization using Optuna with GAT/GATv2

Goal: Find optimal hyperparameters to reach 95%+ F1
Approach: Systematic search with Optuna
Architecture: GAT family (GAT, GATv2, etc.)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pathlib import Path
import json
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt

class ConfigurableGAT(torch.nn.Module):
    """Flexible GAT that can use GAT or GATv2"""
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_heads=8, num_layers=3, dropout=0.3, use_gatv2=False):
        super().__init__()
        
        ConvLayer = GATv2Conv if use_gatv2 else GATConv
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(ConvLayer(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        self.convs.append(ConvLayer(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def split_data(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def train_and_evaluate(trial, data, device):
    """Train model with given hyperparameters and return validation F1"""
    
    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 192, 384, step=32)  # 192, 224, 256, 288, 320, 352, 384
    num_heads = trial.suggest_int('num_heads', 4, 12, step=2)  # 4, 6, 8, 10, 12
    num_layers = trial.suggest_int('num_layers', 3, 4)  # 3 or 4
    dropout = trial.suggest_float('dropout', 0.2, 0.4, step=0.05)  # 0.2, 0.25, 0.3, 0.35, 0.4
    lr = trial.suggest_float('lr', 0.0005, 0.002, log=True)  # Log scale
    weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.001, log=True)
    use_gatv2 = trial.suggest_categorical('use_gatv2', [False, True])
    
    # Initialize model
    model = ConfigurableGAT(
        in_channels=data.x.size(1),
        hidden_channels=hidden_dim,
        out_channels=2,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_gatv2=use_gatv2
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    patience = 15
    max_epochs = 100
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
            y_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(y_true, pred, average='weighted')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
        # Report intermediate value for pruning
        trial.report(val_f1, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_f1


def objective(trial, data_path, device):
    """Objective function for Optuna"""
    
    # Load data
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    data = split_data(data, seed=42)  # Fixed seed for fair comparison
    
    # Train and get validation F1
    val_f1 = train_and_evaluate(trial, data, device)
    
    return val_f1


def main():
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA - TARGET: 95% F1")
    print("="*80)
    
    data_path = "data/graphs_full/graph_data_enriched.pt"
    output_dir = Path("experiments/optuna_hpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cpu')  # Use CPU for stability
    
    print("\n📊 Search Space:")
    print("  hidden_dim: [192, 224, 256, 288, 320, 352, 384]")
    print("  num_heads: [4, 6, 8, 10, 12]")
    print("  num_layers: [3, 4]")
    print("  dropout: [0.2, 0.25, 0.3, 0.35, 0.4]")
    print("  lr: [0.0005, 0.002] (log scale)")
    print("  weight_decay: [0.0001, 0.001] (log scale)")
    print("  use_gatv2: [False, True]")
    
    n_trials = 50
    print(f"\n🔍 Running {n_trials} trials...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, data_path, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # Best trial
    best_trial = study.best_trial
    print(f"\n🏆 Best Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Validation F1: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    
    print(f"\n⚙️  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial_number': best_trial.number,
        'best_val_f1': best_trial.value,
        'best_params': best_trial.params,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ],
        'baseline_f1': 0.9176,
        'target_f1': 0.95
    }
    
    with open(output_dir / 'hpo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir / 'hpo_results.json'}")
    
    # Visualizations
    print("\n📊 Creating visualizations...")
    
    # Optimization history
    fig1 = plot_optimization_history(study)
    fig1.write_image(str(output_dir / 'optimization_history.png'))
    print(f"  ✓ Saved: optimization_history.png")
    
    # Parameter importances
    fig2 = plot_param_importances(study)
    fig2.write_image(str(output_dir / 'param_importances.png'))
    print(f"  ✓ Saved: param_importances.png")
    
    # Now train final model with best parameters on full training data
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    data = split_data(data, seed=42)
    
    best_params = best_trial.params
    
    model = ConfigurableGAT(
        in_channels=data.x.size(1),
        hidden_channels=best_params['hidden_dim'],
        out_channels=2,
        num_heads=best_params['num_heads'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        use_gatv2=best_params['use_gatv2']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    # Train with more epochs for final model
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n🏋️ Training final model...")
    for epoch in range(150):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            
            val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='weighted')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'params': best_params
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= 25:
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Val F1 = {val_f1:.4f}, Best = {best_val_f1:.4f}")
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        
        test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        
        test_acc = accuracy_score(test_true, test_pred)
        test_precision = precision_score(test_true, test_pred, average='weighted')
        test_recall = recall_score(test_true, test_pred, average='weighted')
        test_f1 = f1_score(test_true, test_pred, average='weighted')
    
    print("\n" + "="*80)
    print("FINAL MODEL RESULTS")
    print("="*80)
    print(f"Best Epoch: {best_epoch}")
    print(f"\nValidation Metrics:")
    print(f"  F1 Score: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%) ⭐")
    
    # Check progress
    baseline_f1 = 0.9176
    target_f1 = 0.95
    improvement = test_f1 - baseline_f1
    
    print("\n" + "="*80)
    print("PROGRESS TOWARD 95% F1 TARGET")
    print("="*80)
    print(f"Baseline F1:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Current F1:   {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"Target F1:    {target_f1:.4f} ({target_f1*100:.2f}%)")
    print(f"Improvement:  {improvement:+.4f} ({improvement*100:+.2f} points)")
    
    if test_f1 >= target_f1:
        print("\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
    else:
        gap = target_f1 - test_f1
        progress = (improvement / (target_f1 - baseline_f1)) * 100
        print(f"\nRemaining gap: {gap:.4f} ({gap*100:.2f} points)")
        print(f"Progress:      {progress:.1f}% of the way there")
    
    print("="*80)
    
    # Save final results
    final_results = {
        'best_params': best_params,
        'best_epoch': best_epoch,
        'val_f1': best_val_f1,
        'test_metrics': {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        },
        'baseline_f1': baseline_f1,
        'target_f1': target_f1,
        'improvement': improvement
    }
    
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Final results saved to {output_dir / 'final_results.json'}")
    print("\n✅ HYPERPARAMETER OPTIMIZATION COMPLETE!")


if __name__ == "__main__":
    main()
