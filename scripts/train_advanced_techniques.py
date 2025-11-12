"""
Train with Enhanced Features + Try Multiple Architectures

1. GATv2 with enhanced features
2. GIN (Graph Isomorphism Network)
3. Advanced training techniques (Focal Loss, DropEdge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GINConv, global_mean_pool
from torch_geometric.utils import dropout_edge
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import json
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EnhancedGATv2(nn.Module):
    """GATv2 with best config from HPO"""
    def __init__(self, in_channels, hidden_channels=256, out_channels=2, 
                 num_heads=10, num_layers=3, dropout=0.25):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GINModel(nn.Module):
    """Graph Isomorphism Network - Often stronger than GAT"""
    def __init__(self, in_channels, hidden_channels=256, out_channels=2, 
                 num_layers=4, dropout=0.25):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.output = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.output(x)
        return x


def train_model(model, data, optimizer, criterion, device, use_dropedge=False, drop_edge_rate=0.1):
    """Train one epoch with optional DropEdge"""
    model.train()
    optimizer.zero_grad()
    
    # Apply DropEdge if enabled
    if use_dropedge and model.training:
        edge_index, _ = dropout_edge(data.edge_index, p=drop_edge_rate)
    else:
        edge_index = data.edge_index
    
    out = model(data.x.to(device), edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, device):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        
        val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
        val_true = data.y[data.val_mask].cpu().numpy()
        val_f1 = f1_score(val_true, val_pred, average='weighted')
        
        test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        test_f1 = f1_score(test_true, test_pred, average='weighted')
        test_acc = accuracy_score(test_true, test_pred)
        test_precision = precision_score(test_true, test_pred, average='weighted')
        test_recall = recall_score(test_true, test_pred, average='weighted')
    
    return val_f1, test_f1, test_acc, test_precision, test_recall


def train_and_evaluate_model(model_name, model, data, device, use_focal_loss=False, use_dropedge=False):
    """Train and evaluate a model"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    print(f"Features: {data.x.shape[1]}")
    print(f"Focal Loss: {use_focal_loss}, DropEdge: {use_dropedge}")
    
    # Setup
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0016, weight_decay=0.000134)
    
    best_val_f1 = 0
    best_test_metrics = None
    best_epoch = 0
    patience_counter = 0
    
    pbar = tqdm(range(150), desc=model_name)
    for epoch in pbar:
        # Train
        loss = train_model(model, data, optimizer, criterion, device, use_dropedge, drop_edge_rate=0.1)
        
        # Evaluate
        val_f1, test_f1, test_acc, test_precision, test_recall = evaluate(model, data, device)
        
        pbar.set_postfix({'loss': f'{loss:.4f}', 'val_f1': f'{val_f1:.4f}', 'best': f'{best_val_f1:.4f}'})
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_test_metrics = {
                'f1': test_f1,
                'accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall
            }
        else:
            patience_counter += 1
        
        if patience_counter >= 30:
            break
    
    pbar.close()
    
    print(f"\n✓ Best Epoch: {best_epoch}")
    print(f"  Validation F1: {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    print(f"  Test F1: {best_test_metrics['f1']:.4f} ({best_test_metrics['f1']*100:.2f}%)")
    print(f"  Test Acc: {best_test_metrics['accuracy']:.4f} ({best_test_metrics['accuracy']*100:.2f}%)")
    
    return best_val_f1, best_test_metrics


def main():
    print("="*80)
    print("ADVANCED TRAINING - ENHANCED FEATURES + GIN + FOCAL LOSS + DROPEDGE")
    print("="*80)
    
    device = torch.device('cpu')
    output_dir = Path("experiments/advanced_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load enhanced data
    print("\n📊 Loading enhanced graph data...")
    data = torch.load("data/graphs_full/graph_data_enriched_with_stats.pt", map_location='cpu', weights_only=False)
    
    # Load splits
    splits = torch.load("experiments/baseline_reproduction/best_splits.pt", map_location='cpu', weights_only=False)
    data.train_mask = splits['train_mask']
    data.val_mask = splits['val_mask']
    data.test_mask = splits['test_mask']
    
    print(f"  Features: {data.x.shape[1]} (original 384 + {data.x.shape[1]-384} graph stats)")
    print(f"  Train: {data.train_mask.sum()}")
    print(f"  Val: {data.val_mask.sum()}")
    print(f"  Test: {data.test_mask.sum()}")
    
    results = {}
    
    # 1. GATv2 with enhanced features (baseline comparison)
    print("\n" + "="*80)
    print("EXPERIMENT 1: GATv2 + Enhanced Features")
    print("="*80)
    model1 = EnhancedGATv2(in_channels=data.x.shape[1]).to(device)
    val_f1, test_metrics = train_and_evaluate_model(
        "GATv2 + Enhanced", model1, data, device, 
        use_focal_loss=False, use_dropedge=False
    )
    results['gatv2_enhanced'] = {'val_f1': val_f1, **test_metrics}
    
    # 2. GATv2 + Enhanced + Focal Loss
    print("\n" + "="*80)
    print("EXPERIMENT 2: GATv2 + Enhanced + Focal Loss")
    print("="*80)
    model2 = EnhancedGATv2(in_channels=data.x.shape[1]).to(device)
    val_f1, test_metrics = train_and_evaluate_model(
        "GATv2 + Focal", model2, data, device,
        use_focal_loss=True, use_dropedge=False
    )
    results['gatv2_focal'] = {'val_f1': val_f1, **test_metrics}
    
    # 3. GATv2 + Enhanced + DropEdge
    print("\n" + "="*80)
    print("EXPERIMENT 3: GATv2 + Enhanced + DropEdge")
    print("="*80)
    model3 = EnhancedGATv2(in_channels=data.x.shape[1]).to(device)
    val_f1, test_metrics = train_and_evaluate_model(
        "GATv2 + DropEdge", model3, data, device,
        use_focal_loss=False, use_dropedge=True
    )
    results['gatv2_dropedge'] = {'val_f1': val_f1, **test_metrics}
    
    # 4. GATv2 + Enhanced + Focal + DropEdge (kitchen sink)
    print("\n" + "="*80)
    print("EXPERIMENT 4: GATv2 + Enhanced + Focal + DropEdge")
    print("="*80)
    model4 = EnhancedGATv2(in_channels=data.x.shape[1]).to(device)
    val_f1, test_metrics = train_and_evaluate_model(
        "GATv2 + All", model4, data, device,
        use_focal_loss=True, use_dropedge=True
    )
    results['gatv2_all'] = {'val_f1': val_f1, **test_metrics}
    
    # 5. GIN with enhanced features
    print("\n" + "="*80)
    print("EXPERIMENT 5: GIN + Enhanced Features")
    print("="*80)
    model5 = GINModel(in_channels=data.x.shape[1], hidden_channels=256, num_layers=4).to(device)
    val_f1, test_metrics = train_and_evaluate_model(
        "GIN + Enhanced", model5, data, device,
        use_focal_loss=False, use_dropedge=False
    )
    results['gin_enhanced'] = {'val_f1': val_f1, **test_metrics}
    
    # 6. GIN + Focal Loss + DropEdge
    print("\n" + "="*80)
    print("EXPERIMENT 6: GIN + Enhanced + Focal + DropEdge")
    print("="*80)
    model6 = GINModel(in_channels=data.x.shape[1], hidden_channels=256, num_layers=4).to(device)
    val_f1, test_metrics = train_and_evaluate_model(
        "GIN + All", model6, data, device,
        use_focal_loss=True, use_dropedge=True
    )
    results['gin_all'] = {'val_f1': val_f1, **test_metrics}
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    baseline_f1 = 0.8856  # Previous best from HPO
    
    print(f"\nBaseline (GATv2, original features): {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print("\nAll experiments:")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for i, (name, metrics) in enumerate(sorted_results, 1):
        improvement = metrics['f1'] - baseline_f1
        symbol = "🏆" if i == 1 else f"{i}."
        print(f"\n{symbol} {name}:")
        print(f"   Test F1: {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%) [{improvement:+.4f} vs baseline]")
        print(f"   Test Acc: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    best_name, best_metrics = sorted_results[0]
    best_improvement = best_metrics['f1'] - baseline_f1
    
    print("\n" + "="*80)
    print("PROGRESS TOWARD 90% F1 TARGET")
    print("="*80)
    print(f"\nPrevious best: {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"New best ({best_name}): {best_metrics['f1']:.4f} ({best_metrics['f1']*100:.2f}%)")
    print(f"Improvement: {best_improvement:+.4f} ({best_improvement*100:+.2f} points)")
    
    if best_metrics['f1'] >= 0.90:
        print("\n🎉🎉🎉 90% F1 ACHIEVED! 🎉🎉🎉")
    else:
        gap = 0.90 - best_metrics['f1']
        print(f"\nGap to 90%: {gap:.4f} ({gap*100:.2f} points)")
        if gap < 0.01:
            print("Very close! Try ensemble or minor tuning.")
        elif gap < 0.02:
            print("Close! Additional techniques may help.")
        else:
            print("Significant gap remaining. May need more advanced methods.")
    
    print("="*80)
    
    # Save results
    results['baseline_f1'] = baseline_f1
    results['best_experiment'] = best_name
    results['best_test_f1'] = best_metrics['f1']
    results['improvement'] = best_improvement
    
    with open(output_dir / 'advanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir / 'advanced_training_results.json'}")
    print("\n✅ ADVANCED TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
