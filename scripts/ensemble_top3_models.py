"""
Ensemble Top-3 Models from HPO - Target: 90% F1

Best single model: 88.56% F1
Expected ensemble: 89-90% F1 (+0.5-1.5 points)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import json
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


def train_model(data, params, device, model_name):
    """Train a single model with given hyperparameters"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    print(f"Params: hidden={params['hidden_dim']}, heads={params['num_heads']}, "
          f"layers={params['num_layers']}, dropout={params['dropout']:.2f}")
    print(f"        lr={params['lr']:.4f}, wd={params['weight_decay']:.6f}, gatv2={params['use_gatv2']}")
    
    model = ConfigurableGAT(
        in_channels=data.x.size(1),
        hidden_channels=params['hidden_dim'],
        out_channels=2,
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        use_gatv2=params['use_gatv2']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    pbar = tqdm(range(150), desc=model_name)
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
            
            # Save checkpoint
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= 30:
            break
    
    pbar.close()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    print(f"✓ Best epoch: {best_epoch}, Val F1: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    
    return model, best_val_f1


def main():
    print("="*80)
    print("ENSEMBLE TOP-3 MODELS - TARGET: 90% F1")
    print("="*80)
    
    data_path = "data/graphs_full/graph_data_enriched.pt"
    splits_path = "experiments/baseline_reproduction/best_splits.pt"
    output_dir = Path("experiments/ensemble_top3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cpu')
    
    # Load data
    print("\n📊 Loading data...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    splits = torch.load(splits_path, map_location='cpu', weights_only=False)
    
    data.train_mask = splits['train_mask']
    data.val_mask = splits['val_mask']
    data.test_mask = splits['test_mask']
    
    print(f"  Train: {data.train_mask.sum()}")
    print(f"  Val: {data.val_mask.sum()}")
    print(f"  Test: {data.test_mask.sum()}")
    
    # Top 3 configurations from HPO
    top_configs = [
        {
            'name': 'Model 1 (GATv2)',
            'params': {
                'hidden_dim': 256,
                'num_heads': 10,
                'num_layers': 3,
                'dropout': 0.25,
                'lr': 0.0016064433134414867,
                'weight_decay': 0.0001336520143949727,
                'use_gatv2': True
            }
        },
        {
            'name': 'Model 2 (GAT)',
            'params': {
                'hidden_dim': 320,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.20,
                'lr': 0.0008932863825087093,
                'weight_decay': 0.0006289899610616476,
                'use_gatv2': False
            }
        },
        {
            'name': 'Model 3 (GAT)',
            'params': {
                'hidden_dim': 320,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.20,
                'lr': 0.0008702764726398078,
                'weight_decay': 0.0006145944834989782,
                'use_gatv2': False
            }
        }
    ]
    
    # Train all models
    models = []
    val_f1_scores = []
    
    for config in top_configs:
        model, val_f1 = train_model(data, config['params'], device, config['name'])
        models.append(model)
        val_f1_scores.append(val_f1)
    
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*80)
    
    individual_results = []
    
    for i, (model, config) in enumerate(zip(models, top_configs), 1):
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            
            test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
            test_true = data.y[data.test_mask].cpu().numpy()
            
            test_f1 = f1_score(test_true, test_pred, average='weighted')
            test_acc = accuracy_score(test_true, test_pred)
        
        print(f"\n{config['name']}:")
        print(f"  Val F1:  {val_f1_scores[i-1]:.4f} ({val_f1_scores[i-1]*100:.2f}%)")
        print(f"  Test F1: {test_f1:.4f} ({test_f1*100:.2f}%)")
        print(f"  Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        individual_results.append({
            'name': config['name'],
            'val_f1': val_f1_scores[i-1],
            'test_f1': test_f1,
            'test_acc': test_acc
        })
    
    # Ensemble predictions (soft voting)
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    
    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            probs = F.softmax(out, dim=1)
            all_probs.append(probs)
    
    # Average probabilities
    ensemble_probs = torch.stack(all_probs).mean(dim=0)
    
    # Validation ensemble
    val_ensemble_pred = ensemble_probs[data.val_mask].argmax(dim=1).cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()
    val_ensemble_f1 = f1_score(val_true, val_ensemble_pred, average='weighted')
    
    # Test ensemble
    test_ensemble_pred = ensemble_probs[data.test_mask].argmax(dim=1).cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    
    test_ensemble_f1 = f1_score(test_true, test_ensemble_pred, average='weighted')
    test_ensemble_acc = accuracy_score(test_true, test_ensemble_pred)
    test_ensemble_precision = precision_score(test_true, test_ensemble_pred, average='weighted')
    test_ensemble_recall = recall_score(test_true, test_ensemble_pred, average='weighted')
    
    print(f"\nEnsemble (Soft Voting):")
    print(f"  Validation F1: {val_ensemble_f1:.4f} ({val_ensemble_f1*100:.2f}%)")
    print(f"\n  Test Metrics:")
    print(f"    Accuracy:  {test_ensemble_acc:.4f} ({test_ensemble_acc*100:.2f}%)")
    print(f"    Precision: {test_ensemble_precision:.4f} ({test_ensemble_precision*100:.2f}%)")
    print(f"    Recall:    {test_ensemble_recall:.4f} ({test_ensemble_recall*100:.2f}%)")
    print(f"    F1 Score:  {test_ensemble_f1:.4f} ({test_ensemble_f1*100:.2f}%) ⭐")
    
    # Compare with best individual
    best_individual_f1 = max([r['test_f1'] for r in individual_results])
    ensemble_gain = test_ensemble_f1 - best_individual_f1
    
    print("\n" + "="*80)
    print("PROGRESS SUMMARY")
    print("="*80)
    
    baseline_f1 = 0.8661
    single_model_f1 = 0.8856  # From HPO
    
    print(f"\nBaseline F1:        {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Best Single Model:  {single_model_f1:.4f} ({single_model_f1*100:.2f}%) [+{(single_model_f1-baseline_f1)*100:.2f} pts]")
    print(f"Best Individual:    {best_individual_f1:.4f} ({best_individual_f1*100:.2f}%) [+{(best_individual_f1-baseline_f1)*100:.2f} pts]")
    print(f"Ensemble:           {test_ensemble_f1:.4f} ({test_ensemble_f1*100:.2f}%) [+{(test_ensemble_f1-baseline_f1)*100:.2f} pts]")
    
    print(f"\nEnsemble gain over best individual: {ensemble_gain:+.4f} ({ensemble_gain*100:+.2f} pts)")
    
    if test_ensemble_f1 >= 0.90:
        print("\n🎉 Achieved 90%+ F1 with ensemble!")
    elif test_ensemble_f1 >= 0.89:
        print("\n✅ Very close! 89%+ F1 achieved!")
    else:
        gap_to_90 = 0.90 - test_ensemble_f1
        print(f"\n📊 Gap to 90%: {gap_to_90:.4f} ({gap_to_90*100:.2f} points)")
    
    print("="*80)
    
    # Save results
    results = {
        'individual_models': individual_results,
        'ensemble': {
            'val_f1': val_ensemble_f1,
            'test_acc': test_ensemble_acc,
            'test_precision': test_ensemble_precision,
            'test_recall': test_ensemble_recall,
            'test_f1': test_ensemble_f1
        },
        'baseline_f1': baseline_f1,
        'single_model_f1': single_model_f1,
        'ensemble_gain': ensemble_gain,
        'total_improvement': test_ensemble_f1 - baseline_f1
    }
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir / 'ensemble_results.json'}")
    print("\n✅ ENSEMBLE COMPLETE!")


if __name__ == "__main__":
    main()
