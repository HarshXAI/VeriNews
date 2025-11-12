"""
FAST Hyperparameter Optimization - Practical approach for 95% F1

Key changes:
1. Faster trials (max 30 epochs per trial instead of 100)
2. Progress feedback during trials
3. Smarter pruning
4. Reduced search space (most promising range)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path
import json
import optuna
from tqdm import tqdm

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


def train_and_evaluate(trial, data, device, trial_num, total_trials):
    """Train model with given hyperparameters and return validation F1"""
    
    # FOCUSED hyperparameter search (most promising range)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 288, 320, 352])
    num_heads = trial.suggest_categorical('num_heads', [6, 8, 10])
    num_layers = trial.suggest_int('num_layers', 3, 4)
    dropout = trial.suggest_float('dropout', 0.25, 0.35, step=0.05)
    lr = trial.suggest_float('lr', 0.0008, 0.0015, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0003, 0.0008, log=True)
    use_gatv2 = trial.suggest_categorical('use_gatv2', [False, True])
    
    print(f"\n{'='*80}")
    print(f"Trial {trial_num}/{total_trials}")
    print(f"{'='*80}")
    print(f"Params: hidden={hidden_dim}, heads={num_heads}, layers={num_layers}, "
          f"dropout={dropout:.2f}, lr={lr:.4f}, wd={weight_decay:.4f}, gatv2={use_gatv2}")
    
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
    
    # FAST training loop
    best_val_f1 = 0
    patience_counter = 0
    patience = 8  # More aggressive early stopping
    max_epochs = 30  # Much fewer epochs per trial
    
    pbar = tqdm(range(max_epochs), desc=f'Trial {trial_num}', leave=False)
    for epoch in pbar:
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
        # Evaluate every 2 epochs (save time)
        if epoch % 2 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x.to(device), data.edge_index.to(device))
                pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
                y_true = data.y[data.val_mask].cpu().numpy()
                val_f1 = f1_score(y_true, pred, average='weighted')
            
            pbar.set_postfix({'val_f1': f'{val_f1:.4f}', 'best': f'{best_val_f1:.4f}'})
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                pbar.close()
                print(f"  Early stop at epoch {epoch}, Best Val F1: {best_val_f1:.4f}")
                break
            
            # Report for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                pbar.close()
                print(f"  Pruned at epoch {epoch}")
                raise optuna.TrialPruned()
    
    pbar.close()
    print(f"  ✓ Completed: Best Val F1 = {best_val_f1:.4f}")
    
    return best_val_f1


def objective(trial, data_path, device, trial_num, total_trials):
    """Objective function for Optuna"""
    
    # Load data (only once per trial)
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    data = split_data(data, seed=42)
    
    # Train and get validation F1
    val_f1 = train_and_evaluate(trial, data, device, trial_num, total_trials)
    
    return val_f1


def main():
    print("="*80)
    print("FAST HYPERPARAMETER OPTIMIZATION - TARGET: 95% F1")
    print("="*80)
    
    data_path = "data/graphs_full/graph_data_enriched.pt"
    output_dir = Path("experiments/optuna_hpo_fast")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cpu')
    
    print("\n📊 FOCUSED Search Space (most promising range):")
    print("  hidden_dim: [256, 288, 320, 352]")
    print("  num_heads: [6, 8, 10]")
    print("  num_layers: [3, 4]")
    print("  dropout: [0.25, 0.30, 0.35]")
    print("  lr: [0.0008, 0.0015] (log scale)")
    print("  weight_decay: [0.0003, 0.0008] (log scale)")
    print("  use_gatv2: [False, True]")
    print("\n⚡ Fast mode: 30 epochs max per trial, aggressive early stopping")
    
    n_trials = 30  # Reduced from 50
    print(f"\n🔍 Running {n_trials} trials (estimated: 1.5-2 hours)...\n")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
            interval_steps=2
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize with progress tracking
    for i in range(n_trials):
        try:
            study.optimize(
                lambda trial: objective(trial, data_path, device, i+1, n_trials),
                n_trials=1,
                show_progress_bar=False
            )
            
            # Print progress
            best_so_far = study.best_value
            print(f"\n  📊 Progress: {i+1}/{n_trials} trials, Best F1 so far: {best_so_far:.4f}\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving results so far...")
            break
    
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
        'n_trials_completed': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'value': t.value if t.value is not None else 0.0,
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
    
    # Train final model with best parameters
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
    
    print("\n🏋️ Training final model (max 100 epochs)...")
    pbar = tqdm(range(100), desc='Final Training')
    for epoch in pbar:
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
            
            val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='weighted')
        
        pbar.set_postfix({'val_f1': f'{val_f1:.4f}', 'best': f'{best_val_f1:.4f}'})
        
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
        
        if patience_counter >= 20:
            break
    
    pbar.close()
    
    # Load best model and evaluate on test
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        
        test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        test_acc = accuracy_score(test_true, test_pred)
        test_precision = precision_score(test_true, test_pred, average='weighted')
        test_recall = recall_score(test_true, test_pred, average='weighted')
        test_f1 = f1_score(test_true, test_pred, average='weighted')
    
    print("\n" + "="*80)
    print("FINAL MODEL RESULTS")
    print("="*80)
    print(f"Best Epoch: {best_epoch}")
    print(f"\nValidation F1: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%) ⭐")
    
    # Progress check
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
        progress = (improvement / (target_f1 - baseline_f1)) * 100 if improvement > 0 else 0
        print(f"\nRemaining gap: {gap:.4f} ({gap*100:.2f} points)")
        if progress > 0:
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
