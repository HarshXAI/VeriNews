"""
Improved GAT Training Script - Target: 95%+ F1

Key improvements over baseline (91.76% F1):
1. Increased model capacity (hidden_dim: 256 → 512) - addresses UNDERFITTING
2. Class weights (3.0 for fake, 1.0 for real) - addresses 75/25 imbalance  
3. GATv2Conv (dynamic attention) - better than GAT
4. Slightly reduced dropout (0.3 → 0.25) - we're underfitting, not overfitting!

Expected improvement: +3-4 F1 points → 94-95% F1
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedGATV2(torch.nn.Module):
    """Improved GAT with higher capacity and GATv2"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=3, dropout=0.25):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        # Layer 1: in → hidden
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        # Middle layers: hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        # Final layer: hidden → out
        self.convs.append(GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def train_epoch(model, data, optimizer, device, class_weights=None):
    """Train for one epoch with class weights"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Use class weights to handle imbalance
    if class_weights is not None:
        loss = F.cross_entropy(
            out[data.train_mask], 
            data.y[data.train_mask].to(device),
            weight=class_weights.to(device)
        )
    else:
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
    probs = F.softmax(out[mask], dim=1).cpu().numpy()[:, 1]
    
    acc = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, pred, average='weighted')
    
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = 0.0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'loss': F.cross_entropy(out[mask], data.y[mask].to(device)).item()
    }


def split_data(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits with fixed seed"""
    torch.manual_seed(seed)
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


def main():
    print("=" * 80)
    print("IMPROVED GAT TRAINING - TARGET: 95%+ F1")
    print("=" * 80)
    
    # Configuration
    data_path = "data/graphs_full/graph_data_enriched.pt"
    output_dir = Path("experiments/improved_gat_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters - IMPROVED
    config = {
        'hidden_dim': 512,        # ⬆️ INCREASED from 256 (address underfitting)
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.25,          # ⬇️ REDUCED from 0.3 (we're underfitting!)
        'lr': 0.001,
        'weight_decay': 0.0005,
        'epochs': 150,            # ⬆️ More epochs (early stopping will handle it)
        'patience': 25,           # ⬆️ More patience
        'seed': 42,
        'class_weight_fake': 3.0, # NEW: Handle 75/25 imbalance
        'class_weight_real': 1.0,
        'architecture': 'GATv2',  # NEW: Upgrade to GATv2
    }
    
    print("\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load data
    print(f"\n📂 Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    print(f"  ✓ Loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Create splits
    data = split_data(data, seed=config['seed'])
    print(f"\n📊 Data splits:")
    print(f"  Train: {data.train_mask.sum()} ({data.train_mask.sum()/data.num_nodes*100:.1f}%)")
    print(f"  Val:   {data.val_mask.sum()} ({data.val_mask.sum()/data.num_nodes*100:.1f}%)")
    print(f"  Test:  {data.test_mask.sum()} ({data.test_mask.sum()/data.num_nodes*100:.1f}%)")
    
    # Check class balance
    train_labels = data.y[data.train_mask]
    num_fake = (train_labels == 0).sum().item()
    num_real = (train_labels == 1).sum().item()
    print(f"\n📈 Class distribution (train set):")
    print(f"  Fake: {num_fake} ({num_fake/(num_fake+num_real)*100:.1f}%)")
    print(f"  Real: {num_real} ({num_real/(num_fake+num_real)*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{num_real/num_fake:.2f}")
    
    # Setup class weights
    class_weights = torch.tensor([config['class_weight_fake'], config['class_weight_real']], dtype=torch.float)
    print(f"\n⚖️  Class weights: Fake={class_weights[0]:.1f}, Real={class_weights[1]:.1f}")
    
    # Initialize model
    device = torch.device('cpu')  # Use CPU for stability
    num_features = data.x.size(1)
    num_classes = 2
    
    model = ImprovedGATV2(
        in_channels=num_features,
        hidden_channels=config['hidden_dim'],
        out_channels=num_classes,
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n🤖 Model initialized:")
    print(f"  Architecture: {config['architecture']}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Device: {device}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)
    
    pbar = tqdm(range(config['epochs']), desc='Training')
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, device, class_weights)
        
        # Evaluate
        val_metrics = evaluate(model, data, data.val_mask, device)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_f1': f"{val_metrics['f1']:.4f}",
            'best_f1': f"{best_val_f1:.4f}"
        })
        
        # Check for improvement
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'config': config
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            print(f"  Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
            break
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    # Load best model
    print(f"\n📥 Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n📊 FINAL EVALUATION:")
    
    train_metrics = evaluate(model, data, data.train_mask, device)
    val_metrics = evaluate(model, data, data.val_mask, device)
    test_metrics = evaluate(model, data, data.test_mask, device)
    
    print("\nTrain Set:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
    print(f"  F1 Score:  {train_metrics['f1']:.4f} ({train_metrics['f1']*100:.2f}%)")
    
    print("\nValidation Set:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {val_metrics['precision']:.4f} ({val_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {val_metrics['recall']:.4f} ({val_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {val_metrics['f1']:.4f} ({val_metrics['f1']*100:.2f}%)")
    print(f"  AUC-ROC:   {val_metrics['auc']:.4f} ({val_metrics['auc']*100:.2f}%)")
    
    print("\nTest Set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%) ⭐")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f} ({test_metrics['auc']*100:.2f}%)")
    
    # Check if we hit target
    target_f1 = 0.95
    baseline_f1 = 0.9176
    improvement = test_metrics['f1'] - baseline_f1
    
    print("\n" + "=" * 80)
    print("PROGRESS TOWARD 95% F1 TARGET")
    print("=" * 80)
    print(f"Baseline F1:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Current F1:   {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print(f"Target F1:    {target_f1:.4f} ({target_f1*100:.2f}%)")
    print(f"Improvement:  {improvement:+.4f} ({improvement*100:+.2f} points)")
    
    if test_metrics['f1'] >= target_f1:
        print("\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
    else:
        gap = target_f1 - test_metrics['f1']
        progress = (improvement / (target_f1 - baseline_f1)) * 100
        print(f"\nRemaining gap: {gap:.4f} ({gap*100:.2f} points)")
        print(f"Progress: {progress:.1f}% of the way there")
    
    # Save results
    results = {
        'config': config,
        'best_epoch': best_epoch,
        'num_params': num_params,
        'train_metrics': {k: float(v) for k, v in train_metrics.items()},
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'history': history,
        'baseline_f1': float(baseline_f1),
        'target_f1': float(target_f1),
        'improvement': float(improvement)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir / 'results.json'}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1
    axes[1].plot(history['val_f1'], label='Val F1', color='green', linewidth=2)
    axes[1].axhline(y=baseline_f1, color='blue', linestyle=':', label=f'Baseline ({baseline_f1:.4f})')
    axes[1].axhline(y=target_f1, color='red', linestyle=':', label=f'Target ({target_f1:.4f})')
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"💾 Training curves saved to {output_dir / 'training_curves.png'}")
    
    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
