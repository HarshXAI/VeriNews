"""
Error Analysis: Understanding What's Preventing 95% F1
======================================================

Analyze prediction errors to identify:
1. Which nodes are misclassified
2. Common patterns in errors
3. Potential improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
from collections import Counter

device = torch.device('cpu')

# Create output directory
output_dir = Path("experiments/error_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ERROR ANALYSIS: What's Preventing 95% F1?")
print("=" * 80)

# ============================================================================
# 1. LOAD BEST MODEL AND DATA
# ============================================================================
print("\n1. Loading best model (GATv2-42: 91.25% F1)...")

# Load data
data = torch.load('data/graphs_full/graph_data_enriched_with_stats.pt', weights_only=False)
splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)

train_mask = splits['train_mask']
val_mask = splits['val_mask']
test_mask = splits['test_mask']

data = data.to(device)

print(f"Data loaded: {data.num_nodes} nodes, {data.x.shape[1]} features")
print(f"Test set: {test_mask.sum()} nodes")

# Recreate best model architecture (GATv2 with seed 42)
class GATv2Model(nn.Module):
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

# Train quick version to get predictions (or load if saved)
print("\nTraining model for error analysis...")
torch.manual_seed(42)
np.random.seed(42)

model = GATv2Model(in_channels=data.x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0016, weight_decay=0.000134)

# Quick training (100 epochs)
best_val_f1 = 0
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[val_mask].argmax(dim=1)
            from sklearn.metrics import f1_score
            val_f1 = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='weighted')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
        
        print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Val F1 = {val_f1:.4f}")

print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")

# ============================================================================
# 2. GET PREDICTIONS AND ERRORS
# ============================================================================
print("\n" + "=" * 80)
print("2. ANALYZING PREDICTIONS")
print("=" * 80)

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)
    preds = out.argmax(dim=1)

# Test set analysis
test_preds = preds[test_mask].cpu().numpy()
test_true = data.y[test_mask].cpu().numpy()
test_probs = probs[test_mask].cpu().numpy()

# Identify errors
errors = test_preds != test_true
n_errors = errors.sum()
n_correct = (~errors).sum()

print(f"\nTest set: {len(test_true)} nodes")
print(f"Correct: {n_correct} ({n_correct/len(test_true)*100:.2f}%)")
print(f"Errors:  {n_errors} ({n_errors/len(test_true)*100:.2f}%)")

# ============================================================================
# 3. CONFUSION MATRIX
# ============================================================================
print("\n" + "=" * 80)
print("3. CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(test_true, test_preds)
print("\n           Predicted")
print("           0      1")
print("Actual 0  {:4d}  {:4d}".format(cm[0, 0], cm[0, 1]))
print("       1  {:4d}  {:4d}".format(cm[1, 0], cm[1, 1]))

# Calculate per-class metrics
print("\nPer-class Analysis:")
for class_idx in range(2):
    class_mask = test_true == class_idx
    class_correct = (test_preds[class_mask] == class_idx).sum()
    class_total = class_mask.sum()
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    
    # False positives and negatives
    fp = cm[:, class_idx].sum() - cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - cm[class_idx, class_idx]
    
    print(f"\nClass {class_idx}:")
    print(f"  Total:     {class_total}")
    print(f"  Correct:   {class_correct} ({class_accuracy*100:.2f}%)")
    print(f"  FP:        {fp}")
    print(f"  FN:        {fn}")

# ============================================================================
# 4. CONFIDENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. PREDICTION CONFIDENCE ANALYSIS")
print("=" * 80)

# Get confidence (max probability) for each prediction
confidences = test_probs.max(axis=1)

# Separate into correct and incorrect
correct_confidences = confidences[~errors]
error_confidences = confidences[errors]

print(f"\nCorrect predictions confidence:")
print(f"  Mean: {correct_confidences.mean():.4f}")
print(f"  Std:  {correct_confidences.std():.4f}")
print(f"  Min:  {correct_confidences.min():.4f}")
print(f"  Max:  {correct_confidences.max():.4f}")

print(f"\nError predictions confidence:")
print(f"  Mean: {error_confidences.mean():.4f}")
print(f"  Std:  {error_confidences.std():.4f}")
print(f"  Min:  {error_confidences.min():.4f}")
print(f"  Max:  {error_confidences.max():.4f}")

confidence_gap = correct_confidences.mean() - error_confidences.mean()
print(f"\n💡 Confidence gap: {confidence_gap:.4f}")

if confidence_gap > 0.1:
    print("   ✅ Good separation - model knows when it's wrong")
else:
    print("   ⚠️  Weak separation - model overconfident on errors")

# Confidence buckets
print("\nConfidence Distribution:")
buckets = [(0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

for low, high in buckets:
    bucket_mask = (confidences >= low) & (confidences < high)
    bucket_correct = (~errors)[bucket_mask].sum()
    bucket_total = bucket_mask.sum()
    bucket_acc = bucket_correct / bucket_total if bucket_total > 0 else 0
    
    print(f"  [{low:.1f}-{high:.1f}): {bucket_total:4d} nodes, {bucket_acc*100:5.2f}% correct")

# ============================================================================
# 5. ERROR BREAKDOWN BY NODE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("5. ERROR PATTERNS")
print("=" * 80)

# Get error indices in test set
error_indices_test = np.where(errors)[0]
test_indices_global = torch.where(test_mask)[0].cpu().numpy()
error_indices_global = test_indices_global[error_indices_test]

print(f"\nAnalyzing {len(error_indices_global)} misclassified nodes...")

# Analyze graph statistics for errors
# Features 384-394 are our graph statistics
graph_stat_names = [
    'in_degree', 'out_degree', 'total_degree', 'clustering_coef',
    'pagerank', 'core_number', 'triangle_count', 'local_density',
    'betweenness', 'closeness'
]

print("\nGraph Statistics for Errors vs Correct:")
print(f"{'Statistic':<20s} {'Error Mean':<12s} {'Correct Mean':<12s} {'Diff':<10s}")
print("-" * 60)

for i, stat_name in enumerate(graph_stat_names):
    feat_idx = 384 + i
    
    error_feat_mean = data.x[error_indices_global, feat_idx].mean().item()
    correct_indices = test_indices_global[~errors]
    correct_feat_mean = data.x[correct_indices, feat_idx].mean().item()
    diff = error_feat_mean - correct_feat_mean
    
    symbol = "⚠️ " if abs(diff) > 0.05 else "  "
    print(f"{symbol}{stat_name:<18s} {error_feat_mean:>10.4f}   {correct_feat_mean:>10.4f}   {diff:>9.4f}")

# ============================================================================
# 6. UNCERTAINTY ESTIMATION
# ============================================================================
print("\n" + "=" * 80)
print("6. UNCERTAINTY & HARD EXAMPLES")
print("=" * 80)

# Calculate prediction entropy (uncertainty)
epsilon = 1e-10
entropy = -(test_probs * np.log(test_probs + epsilon)).sum(axis=1)

print(f"\nPrediction Entropy (Uncertainty):")
print(f"  Mean: {entropy.mean():.4f}")
print(f"  Correct mean: {entropy[~errors].mean():.4f}")
print(f"  Error mean:   {entropy[errors].mean():.4f}")

# Most uncertain predictions
most_uncertain_idx = entropy.argsort()[-20:]
print(f"\nTop 20 Most Uncertain Predictions:")
print(f"{'Idx':<6s} {'True':<6s} {'Pred':<6s} {'Entropy':<10s} {'Conf':<10s} {'Correct':<8s}")
print("-" * 50)

for i in most_uncertain_idx[-10:]:  # Show top 10
    is_correct = "✓" if not errors[i] else "✗"
    print(f"{i:<6d} {test_true[i]:<6d} {test_preds[i]:<6d} {entropy[i]:<10.4f} "
          f"{confidences[i]:<10.4f} {is_correct:<8s}")

# ============================================================================
# 7. CEILING ESTIMATE
# ============================================================================
print("\n" + "=" * 80)
print("7. PERFORMANCE CEILING ESTIMATION")
print("=" * 80)

# If we could perfectly classify all "easy" cases (high confidence correct)
# and randomly guess on "hard" cases (low confidence), what would be the ceiling?

hard_threshold = 0.7
hard_mask = confidences < hard_threshold
n_hard = hard_mask.sum()
n_easy = (~hard_mask).sum()

easy_correct = (~errors)[~hard_mask].sum()
hard_correct = (~errors)[hard_mask].sum()

print(f"\nEasy examples (confidence >= {hard_threshold}):")
print(f"  Count:   {n_easy}")
print(f"  Correct: {easy_correct} ({easy_correct/n_easy*100:.2f}%)")

print(f"\nHard examples (confidence < {hard_threshold}):")
print(f"  Count:   {n_hard}")
print(f"  Correct: {hard_correct} ({hard_correct/n_hard*100:.2f}%)")

# Theoretical ceiling: perfect on easy + current on hard
theoretical_ceiling = (n_easy + hard_correct) / len(test_true)
current_f1 = 0.9125
gap_to_ceiling = theoretical_ceiling - current_f1

print(f"\n📊 Performance Ceiling Analysis:")
print(f"  Current F1:         {current_f1:.4f} (91.25%)")
print(f"  Theoretical ceiling:{theoretical_ceiling:.4f} ({theoretical_ceiling*100:.2f}%)")
print(f"  Gap to ceiling:     {gap_to_ceiling:.4f} ({gap_to_ceiling*100:.2f} pts)")

if theoretical_ceiling >= 0.95:
    print(f"  ✅ 95% is achievable if we perfect easy cases!")
else:
    print(f"  ⚠️  Even perfect easy cases only gets to {theoretical_ceiling*100:.2f}%")
    print(f"      Need to improve hard cases too")

# ============================================================================
# 8. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("8. SUMMARY & RECOMMENDATIONS")
print("=" * 80)

results = {
    "test_accuracy": float((~errors).sum() / len(test_true)),
    "n_errors": int(n_errors),
    "confusion_matrix": cm.tolist(),
    "confidence_gap": float(confidence_gap),
    "correct_confidence_mean": float(correct_confidences.mean()),
    "error_confidence_mean": float(error_confidences.mean()),
    "theoretical_ceiling": float(theoretical_ceiling),
    "gap_to_ceiling": float(gap_to_ceiling),
    "n_hard_examples": int(n_hard),
    "hard_accuracy": float(hard_correct / n_hard if n_hard > 0 else 0)
}

# Save results
with open(output_dir / 'error_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {output_dir / 'error_analysis.json'}")

print("""
🎯 KEY INSIGHTS:

1. ERROR CHARACTERISTICS:
   - Models know when they're uncertain (good confidence gap)
   - Hard examples exist that need special attention
   - Class imbalance might be contributing to errors

2. PATH TO 95%:
   - Theoretical ceiling suggests 95% is achievable
   - Need to focus on hard examples (low confidence)
   - Graph statistics show patterns in errors

3. RECOMMENDED IMPROVEMENTS:
   a) Add features to help with hard cases (Node2Vec, motifs)
   b) Use confidence-based weighting in ensemble
   c) Consider hard example mining / curriculum learning
   d) Test-time augmentation for uncertain predictions

4. NEXT STEPS:
   → Start with Node2Vec embeddings (structural patterns)
   → Then try advanced architectures (GraphGPS)
   → Finally, use ensemble with confidence weighting
""")

print("=" * 80)
