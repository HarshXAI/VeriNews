"""
Hyperparameter optimization for GAT using Optuna
Supports ASHA for efficient trial pruning
"""

import argparse
import os
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from improved training script
from train_gat_improved import (
    ImprovedGATv2, FocalLoss, create_temporal_splits,
    train_epoch_minibatch, evaluate_minibatch
)
from src.utils import set_seed, get_device


def objective(trial, args, data, device):
    """Optuna objective function"""
    
    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    num_heads = trial.suggest_categorical('num_heads', [4, 6, 8])
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    drop_edge_p = trial.suggest_float('drop_edge_p', 0.1, 0.3)
    lr = trial.suggest_float('lr', 2e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 2e-3, log=True)
    
    # Focal loss parameters
    if args.loss == 'focal':
        focal_alpha = trial.suggest_float('focal_alpha', 0.2, 0.4)
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 2.5)
    else:
        focal_alpha = 0.25
        focal_gamma = 2.0
    
    # Create splits (use same seed for reproducibility)
    set_seed(args.seed)
    
    if args.split_method == "temporal":
        import numpy as np
        timestamps = np.arange(data.num_nodes)  # Proxy
        train_mask, val_mask, _ = create_temporal_splits(
            data, timestamps, args.train_ratio, args.val_ratio
        )
    else:
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_size = int(args.train_ratio * num_nodes)
        val_size = int(args.val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
    
    # Create loaders
    from torch_geometric.loader import NeighborLoader
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=train_mask,
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=val_mask,
        shuffle=False,
    )
    
    # Create model
    model = ImprovedGATv2(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=2,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        edge_dim=None,
        use_residual=True,
        drop_edge_p=drop_edge_p
    ).to(device)
    
    # Loss and optimizer
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, args.max_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch_minibatch(
            model, train_loader, optimizer, device, criterion, False
        )
        
        # Validate
        val_metrics = evaluate_minibatch(
            model, val_loader, device, criterion, False
        )
        
        # Track best
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report to Optuna for pruning
        trial.report(val_metrics['f1'], epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if patience_counter >= args.patience:
            break
    
    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    
    # Data
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/optuna_study")
    
    # Optimization
    parser.add_argument("--n-trials", type=int, default=60,
                       help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Timeout in seconds")
    parser.add_argument("--study-name", type=str, default="gat_optimization")
    
    # Training config
    parser.add_argument("--max-epochs", type=int, default=100,
                       help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-neighbors", type=int, nargs='+', default=[15, 10, 5])
    
    # Data splits
    parser.add_argument("--split-method", type=str, default="temporal",
                       choices=["random", "temporal"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    
    # Loss
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    
    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Parallel trials (experimental)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*70)
    
    device = get_device() if args.device == "auto" else args.device
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    data = torch.load(args.data_path, weights_only=False)
    print(f"  ✓ Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    
    # Create study
    print(f"\n🔬 Creating Optuna study...")
    print(f"  • Trials: {args.n_trials}")
    print(f"  • Max epochs per trial: {args.max_epochs}")
    print(f"  • Pruner: MedianPruner")
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Optimize
    print("\n" + "="*70)
    print("OPTIMIZATION")
    print("="*70)
    
    study.optimize(
        lambda trial: objective(trial, args, data, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\n🏆 Best trial:")
    print(f"  • Trial: {study.best_trial.number}")
    print(f"  • F1 Score: {study.best_trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    • {key}: {value}")
    
    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_f1': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'study_name': args.study_name
    }
    
    with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save study
    import joblib
    joblib.dump(study, os.path.join(args.output_dir, 'study.pkl'))
    
    print(f"\n📁 Saved to: {args.output_dir}/")
    
    # Top 5 trials
    print(f"\n📈 Top 5 trials:")
    df = study.trials_dataframe()
    df_sorted = df.sort_values('value', ascending=False).head(5)
    print(df_sorted[['number', 'value', 'params_hidden_dim', 'params_num_layers', 
                     'params_num_heads', 'params_dropout', 'params_lr']])
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
