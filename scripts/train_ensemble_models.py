"""
Ensemble Learning Approach - Train multiple models and combine predictions

Strategy: Train 5 baseline models with different random seeds, then ensemble
their predictions using soft voting (average probabilities).

This is a conservative, proven approach that typically gains +0.5 to +1.5 F1 points.

Baseline: 91.76% F1
Target: 95.00% F1
Expected with Ensemble: 92.5-93.5% F1
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification - EXACT baseline architecture"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def split_data(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits with specific seed"""
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


def train_epoch(model, data, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask].to(device)).sum().item() / data.train_mask.sum().item()
    
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model on given mask"""
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    
    pred = out[mask].argmax(dim=1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    
    # Get probabilities for AUC
    probs = F.softmax(out[mask], dim=1).cpu().numpy()
    
    acc = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, pred, average='weighted')
    
    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except:
        auc = 0.0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'probs': probs
    }


def train_single_model(seed, data_path, output_dir, config):
    """Train a single model with given seed"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL #{seed} (Seed: {seed})")
    print(f"{'='*80}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load and split data
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    data = split_data(data, seed=seed)
    
    print(f"📊 Data splits (seed {seed}):")
    print(f"  Train: {data.train_mask.sum()}")
    print(f"  Val:   {data.val_mask.sum()}")
    print(f"  Test:  {data.test_mask.sum()}")
    
    # Initialize model
    device = torch.device('cpu')
    num_features = data.x.size(1)
    num_classes = 2
    
    model = SimpleGATNode(
        in_channels=num_features,
        hidden_channels=config['hidden_dim'],
        out_channels=num_classes,
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    pbar = tqdm(range(config['epochs']), desc=f'Model {seed}')
    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, data, optimizer, device)
        val_metrics = evaluate(model, data, data.val_mask, device)
        
        pbar.set_postfix({
            'val_f1': f"{val_metrics['f1']:.4f}",
            'best': f"{best_val_f1:.4f}"
        })
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            model_path = output_dir / f'model_seed_{seed}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'seed': seed
            }, model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    # Load best model and evaluate
    checkpoint = torch.load(output_dir / f'model_seed_{seed}.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, data, data.test_mask, device)
    
    print(f"\n✅ Model {seed} Complete:")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Val F1:  {best_val_f1:.4f}")
    print(f"   Test F1: {test_metrics['f1']:.4f}")
    
    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'val_f1': best_val_f1,
        'test_f1': test_metrics['f1'],
        'test_metrics': {k: v for k, v in test_metrics.items() if k != 'probs'},
        'model_path': str(output_dir / f'model_seed_{seed}.pt'),
        'data_masks': {
            'train_mask': data.train_mask,
            'val_mask': data.val_mask,
            'test_mask': data.test_mask
        }
    }


def ensemble_predictions(models_info, data_path, output_dir, config):
    """Ensemble predictions from multiple models using soft voting"""
    
    print(f"\n{'='*80}")
    print("ENSEMBLE PREDICTIONS")
    print(f"{'='*80}")
    
    device = torch.device('cpu')
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    num_models = len(models_info)
    print(f"\n🔮 Ensembling {num_models} models...")
    
    # Collect predictions from all models
    all_probs = []
    
    for info in models_info:
        seed = info['seed']
        print(f"  Loading model (seed {seed})...")
        
        # Recreate data split for this model
        data_split = split_data(torch.load(data_path, map_location='cpu', weights_only=False), seed=seed)
        
        # Load model
        model = SimpleGATNode(
            in_channels=data.x.size(1),
            hidden_channels=config['hidden_dim'],
            out_channels=2,
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        checkpoint = torch.load(info['model_path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            out = model(data_split.x.to(device), data_split.edge_index.to(device))
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    # Average probabilities (soft voting)
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # Evaluate ensemble on each model's test set
    print("\n📊 Evaluating ensemble on each model's test split:")
    ensemble_results = []
    
    for info in models_info:
        seed = info['seed']
        test_mask = info['data_masks']['test_mask'].numpy()
        
        y_true = data.y.numpy()[test_mask]
        y_pred = ensemble_preds[test_mask]
        y_probs = ensemble_probs[test_mask, 1]
        
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_probs)
        
        ensemble_results.append({
            'seed': seed,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })
        
        print(f"  Seed {seed}: F1 = {f1:.4f} (individual: {info['test_f1']:.4f}, gain: {(f1-info['test_f1'])*100:+.2f} points)")
    
    # Average ensemble performance
    avg_ensemble_f1 = np.mean([r['f1'] for r in ensemble_results])
    avg_individual_f1 = np.mean([info['test_f1'] for info in models_info])
    
    print(f"\n{'='*80}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*80}")
    print(f"Average Individual F1: {avg_individual_f1:.4f} ({avg_individual_f1*100:.2f}%)")
    print(f"Average Ensemble F1:   {avg_ensemble_f1:.4f} ({avg_ensemble_f1*100:.2f}%)")
    print(f"Ensemble Gain:         {(avg_ensemble_f1-avg_individual_f1)*100:+.2f} points")
    
    return {
        'individual_results': models_info,
        'ensemble_results': ensemble_results,
        'avg_individual_f1': avg_individual_f1,
        'avg_ensemble_f1': avg_ensemble_f1,
        'ensemble_gain': avg_ensemble_f1 - avg_individual_f1
    }


def main():
    print("="*80)
    print("ENSEMBLE TRAINING - 5 MODELS WITH DIFFERENT SEEDS")
    print("="*80)
    
    # Configuration - EXACT baseline settings
    config = {
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.3,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'epochs': 100,
        'patience': 20,
    }
    
    data_path = "data/graphs_full/graph_data_enriched.pt"
    output_dir = Path("experiments/ensemble_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Seeds for 5 different models
    seeds = [42, 123, 456, 789, 2024]
    
    print("\n⚙️  Configuration (Baseline Architecture):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎲 Training {len(seeds)} models with seeds: {seeds}")
    
    # Train individual models
    models_info = []
    for seed in seeds:
        info = train_single_model(seed, data_path, output_dir, config)
        models_info.append(info)
    
    # Ensemble predictions
    ensemble_results = ensemble_predictions(models_info, data_path, output_dir, config)
    
    # Save results
    results = {
        'config': config,
        'seeds': seeds,
        'num_models': len(seeds),
        'individual_results': [
            {k: v for k, v in info.items() if k != 'data_masks'} 
            for info in models_info
        ],
        'ensemble_summary': {
            'avg_individual_f1': ensemble_results['avg_individual_f1'],
            'avg_ensemble_f1': ensemble_results['avg_ensemble_f1'],
            'ensemble_gain': ensemble_results['ensemble_gain']
        },
        'ensemble_results_per_seed': ensemble_results['ensemble_results'],
        'baseline_f1': 0.9176,
        'target_f1': 0.95
    }
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir / 'ensemble_results.json'}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual vs Ensemble F1
    individual_f1s = [info['test_f1'] for info in models_info]
    ensemble_f1s = [r['f1'] for r in ensemble_results['ensemble_results']]
    
    x = np.arange(len(seeds))
    width = 0.35
    
    axes[0].bar(x - width/2, individual_f1s, width, label='Individual', alpha=0.8)
    axes[0].bar(x + width/2, ensemble_f1s, width, label='Ensemble', alpha=0.8)
    axes[0].axhline(y=0.9176, color='blue', linestyle=':', label='Baseline (91.76%)', linewidth=2)
    axes[0].axhline(y=0.95, color='red', linestyle=':', label='Target (95.00%)', linewidth=2)
    axes[0].set_xlabel('Model (Seed)', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('Individual vs Ensemble Performance', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot comparison
    data_to_plot = [individual_f1s, ensemble_f1s]
    axes[1].boxplot(data_to_plot, labels=['Individual', 'Ensemble'])
    axes[1].axhline(y=0.9176, color='blue', linestyle=':', label='Baseline', linewidth=2)
    axes[1].axhline(y=0.95, color='red', linestyle=':', label='Target', linewidth=2)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Distribution of F1 Scores', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved to {output_dir / 'ensemble_comparison.png'}")
    
    # Final summary
    baseline_f1 = 0.9176
    target_f1 = 0.95
    avg_ensemble_f1 = ensemble_results['avg_ensemble_f1']
    
    print("\n" + "="*80)
    print("PROGRESS TOWARD 95% F1 TARGET")
    print("="*80)
    print(f"Baseline F1:        {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Ensemble F1:        {avg_ensemble_f1:.4f} ({avg_ensemble_f1*100:.2f}%)")
    print(f"Target F1:          {target_f1:.4f} ({target_f1*100:.2f}%)")
    print(f"Improvement:        {(avg_ensemble_f1-baseline_f1)*100:+.2f} points")
    
    if avg_ensemble_f1 >= target_f1:
        print("\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
    else:
        gap = target_f1 - avg_ensemble_f1
        progress = ((avg_ensemble_f1 - baseline_f1) / (target_f1 - baseline_f1)) * 100
        print(f"\nRemaining gap:      {gap:.4f} ({gap*100:.2f} points)")
        print(f"Progress:           {progress:.1f}% of the way there")
        
        if progress > 50:
            print("\n💡 Next step: Combine ensemble with hyperparameter optimization!")
        else:
            print("\n💡 Next step: Try hyperparameter optimization on ensemble members!")
    
    print("="*80)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
