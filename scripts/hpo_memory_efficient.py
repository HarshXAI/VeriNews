"""
Memory-Efficient HPO - Focus on promising configurations
Avoid memory-heavy models, target realistic improvements
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import json
import optuna
from tqdm import tqdm

class ConfigurableGAT(torch.nn.Module):
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


def train_and_evaluate(trial, data, device, trial_num, total_trials):
    """Train model and return validation F1"""
    
    # MEMORY-EFFICIENT search space (avoid 512-dim + 12-heads combinations)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 320, 384])
    num_heads = trial.suggest_categorical('num_heads', [6, 8, 10])
    num_layers = trial.suggest_int('num_layers', 3, 4)
    dropout = trial.suggest_float('dropout', 0.2, 0.35, step=0.05)
    lr = trial.suggest_float('lr', 0.0008, 0.0025, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.0008, log=True)
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
    
    best_val_f1 = 0
    patience_counter = 0
    patience = 15
    max_epochs = 80
    
    pbar = tqdm(range(max_epochs), desc=f'Trial {trial_num}', leave=False)
    for epoch in pbar:
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
        # Evaluate
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
    if patience_counter < patience:
        print(f"  ✓ Completed: Best Val F1 = {best_val_f1:.4f}")
    
    return best_val_f1


def main():
    print("="*80)
    print("MEMORY-EFFICIENT HPO - TARGET: 88-90% F1")
    print("="*80)
    
    data_path = "data/graphs_full/graph_data_enriched.pt"
    splits_path = "experiments/baseline_reproduction/best_splits.pt"
    output_dir = Path("experiments/hpo_memory_efficient")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cpu')
    
    # Load data with FIXED splits
    print("\n📊 Loading data with fixed splits (seed 314)...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    splits = torch.load(splits_path, map_location='cpu', weights_only=False)
    
    data.train_mask = splits['train_mask']
    data.val_mask = splits['val_mask']
    data.test_mask = splits['test_mask']
    
    print(f"  Train: {data.train_mask.sum()} ({data.y[data.train_mask].float().mean():.3f} positive)")
    print(f"  Val: {data.val_mask.sum()} ({data.y[data.val_mask].float().mean():.3f} positive)")
    print(f"  Test: {data.test_mask.sum()} ({data.y[data.test_mask].float().mean():.3f} positive)")
    
    print("\n📊 MEMORY-EFFICIENT Search Space:")
    print("  hidden_dim: [256, 320, 384]")
    print("  num_heads: [6, 8, 10]")
    print("  num_layers: [3, 4]")
    print("  dropout: [0.2, 0.25, 0.3, 0.35]")
    print("  lr: [0.0008, 0.0025] (log scale)")
    print("  weight_decay: [0.0001, 0.0008] (log scale)")
    print("  use_gatv2: [False, True]")
    
    n_trials = 40
    print(f"\n🔍 Running {n_trials} trials...")
    print(f"Baseline: 86.61% F1 (seed 314)")
    print(f"Target: 88-90% F1 (+1.5-3.5 points)\n")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=8,
            interval_steps=2
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    for i in range(n_trials):
        try:
            study.optimize(
                lambda trial: train_and_evaluate(trial, data, device, i+1, n_trials),
                n_trials=1,
                show_progress_bar=False
            )
            
            best_so_far = study.best_value
            print(f"\n  📊 Progress: {i+1}/{n_trials} trials, Best Val F1: {best_so_far:.4f} ({best_so_far*100:.2f}%)\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving results...")
            break
        except Exception as e:
            print(f"\n⚠️  Error in trial {i+1}: {e}")
            print("Continuing with next trial...\n")
            continue
    
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
    
    # Save HPO results
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
        'baseline_f1': 0.8661,
        'target_f1': 0.90
    }
    
    with open(output_dir / 'hpo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir / 'hpo_results.json'}")
    
    # Train final model
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)
    
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
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n🏋️ Training final model (max 150 epochs)...")
    pbar = tqdm(range(150), desc='Final Training')
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
        
        if patience_counter >= 30:
            break
    
    pbar.close()
    
    # Load best and test
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
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best Epoch: {best_epoch}")
    print(f"\nValidation F1: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%) ⭐")
    
    baseline_f1 = 0.8661
    improvement = test_f1 - baseline_f1
    
    print("\n" + "="*80)
    print("PROGRESS")
    print("="*80)
    print(f"Baseline F1:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Optimized F1: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"Improvement:  {improvement:+.4f} ({improvement*100:+.2f} points)")
    
    if test_f1 >= 0.90:
        print("\n🎉 Excellent! Reached 90%+ F1!")
    elif test_f1 >= 0.88:
        print("\n✅ Good progress! 88%+ achieved.")
    elif improvement > 0:
        print(f"\n📈 Positive improvement of {improvement*100:.2f} points!")
    else:
        print("\n⚠️  No improvement over baseline. May need advanced techniques.")
    
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
        'improvement': improvement,
        'split_seed': 314
    }
    
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Final results saved to {output_dir / 'final_results.json'}")
    print("\n✅ COMPLETE!")


if __name__ == "__main__":
    main()
