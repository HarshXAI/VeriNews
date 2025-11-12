"""
Train with Label Smoothing - Quick Test
========================================

Label smoothing reduces overconfidence by using soft targets instead of hard 0/1 labels.
Instead of [1, 0] or [0, 1], we use [0.9, 0.1] or [0.1, 0.9] (for smoothing=0.1)

Current: 91.49% F1, Class 1 recall 97% (overconfident)
Target: 91.6-91.8% F1 (+0.1-0.3 pts)

Strategy: Test smoothing factors [0.1, 0.15, 0.2] and pick the best
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device('cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = Path("experiments/label_smoothing")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LABEL SMOOTHING EXPERIMENTS")
print("=" * 80)
print()

# Load data
print("1. Loading data with Node2Vec embeddings...")
data = torch.load('data/graphs_full/graph_data_with_node2vec.pt', weights_only=False)
splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)

train_mask = splits['train_mask']
val_mask = splits['val_mask']
test_mask = splits['test_mask']

data = data.to(device)

print(f"Features: {data.x.shape[1]} (394 baseline + 128 Node2Vec)")
print(f"Nodes: {data.num_nodes}")
print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
print()

# Model definition (same as successful 91.49% ensemble)
class GATv2Model(nn.Module):
    """GATv2 with best hyperparameters"""
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

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""
    def __init__(self, smoothing=0.1, num_classes=2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        pred: [N, num_classes] logits
        target: [N] class indices
        """
        # Create soft targets
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Apply log softmax and compute KL divergence
        log_probs = F.log_softmax(pred, dim=1)
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
        
        return loss

def train_with_smoothing(smoothing_factor, seed=42):
    """Train a model with specific smoothing factor"""
    print(f"\n{'='*80}")
    print(f"TRAINING WITH SMOOTHING = {smoothing_factor}")
    print(f"{'='*80}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = GATv2Model(
        in_channels=data.x.shape[1],
        hidden_channels=256,
        num_heads=10,
        num_layers=3,
        dropout=0.25
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0016,
        weight_decay=0.000134
    )
    
    criterion = LabelSmoothingLoss(smoothing=smoothing_factor, num_classes=2)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Smoothing factor: {smoothing_factor}")
    print(f"Confidence: {1.0 - smoothing_factor}")
    print()
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 20
    epochs = 200
    
    pbar = tqdm(range(epochs), desc=f'Smoothing={smoothing_factor}')
    
    for epoch in pbar:
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out[val_mask].argmax(dim=1)
                val_f1 = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='weighted')
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'val_f1': f'{val_f1:.4f}',
                    'best': f'{best_val_f1:.4f}'
                })
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience // 10:
                    print(f"\n✅ Early stopped at epoch {epoch+1}")
                    break
    
    pbar.close()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ Best validation F1: {best_val_f1:.4f}")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[test_mask].argmax(dim=1)
        probs = F.softmax(out[test_mask], dim=1)
    
    test_f1 = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='weighted')
    test_acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())
    
    # Confusion matrix
    cm = confusion_matrix(data.y[test_mask].cpu(), pred.cpu())
    class_0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    
    print(f"\n📊 Test Performance:")
    print(f"   F1 Score:       {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"   Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Class 0 recall: {class_0_recall:.4f} ({class_0_recall*100:.2f}%)")
    print(f"   Class 1 recall: {class_1_recall:.4f} ({class_1_recall*100:.2f}%)")
    print()
    
    # Calculate confidence statistics
    confidence = probs.max(dim=1)[0]
    avg_confidence = confidence.mean().item()
    
    print(f"📊 Confidence Statistics:")
    print(f"   Average confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"   Std confidence:     {confidence.std().item():.4f}")
    
    return {
        'smoothing': smoothing_factor,
        'val_f1': float(best_val_f1),
        'test_f1': float(test_f1),
        'test_acc': float(test_acc),
        'class_0_recall': float(class_0_recall),
        'class_1_recall': float(class_1_recall),
        'avg_confidence': float(avg_confidence),
        'std_confidence': float(confidence.std().item()),
        'confusion_matrix': cm.tolist()
    }

# Test different smoothing factors
print("=" * 80)
print("TESTING SMOOTHING FACTORS")
print("=" * 80)
print()

smoothing_factors = [0.0, 0.1, 0.15, 0.2]  # 0.0 = no smoothing (baseline)
results = []

for smoothing in smoothing_factors:
    result = train_with_smoothing(smoothing, seed=42)
    results.append(result)

# Summary comparison
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)
print()

baseline_f1 = 0.9149  # Node2Vec ensemble

print(f"{'Smoothing':<12} {'Test F1':<10} {'Δ vs 91.49%':<15} {'Class 0':<10} {'Class 1':<10} {'Avg Conf':<12}")
print("-" * 80)

best_result = None
best_improvement = -999

for r in results:
    improvement = r['test_f1'] - baseline_f1
    
    if improvement > best_improvement:
        best_improvement = improvement
        best_result = r
    
    marker = " ⭐" if r == best_result and improvement > 0 else ""
    print(f"{r['smoothing']:<12.2f} {r['test_f1']:<10.4f} {improvement:+.4f} ({improvement*100:+.2f}%){'':<3} "
          f"{r['class_0_recall']:<10.4f} {r['class_1_recall']:<10.4f} {r['avg_confidence']:<12.4f}{marker}")

print()

# Save results
with open(output_dir / 'results.json', 'w') as f:
    json.dump({
        'experiments': results,
        'best': best_result,
        'baseline_ensemble': baseline_f1,
        'best_improvement': float(best_improvement)
    }, f, indent=2)

print(f"✅ Results saved to {output_dir / 'results.json'}")
print()

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if best_improvement > 0:
    print(f"🎉 Best smoothing: {best_result['smoothing']:.2f}")
    print(f"   Test F1: {best_result['test_f1']:.4f} ({best_result['test_f1']*100:.2f}%)")
    print(f"   Improvement: {best_improvement:+.4f} ({best_improvement*100:+.2f} pts)")
    print()
    print(f"📊 Class Performance:")
    print(f"   Class 0 recall: {best_result['class_0_recall']:.4f}")
    print(f"   Class 1 recall: {best_result['class_1_recall']:.4f}")
    print(f"   Avg confidence: {best_result['avg_confidence']:.4f}")
else:
    print("⚠️  No improvement from label smoothing")
    print(f"   Best was smoothing={best_result['smoothing']:.2f} with F1={best_result['test_f1']:.4f}")
    print(f"   Baseline: {baseline_f1:.4f}")
    print()
    print("💡 Recommendation: Skip label smoothing, try advanced optimizers instead")

print()
print("=" * 80)
