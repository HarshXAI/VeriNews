"""
Optimize classification threshold for best F1 score.

Current model uses default 0.5 threshold. This script finds the optimal
threshold on validation set that maximizes F1 score.

Expected improvement: +0.5 to +1.5 F1 points
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_curve, auc
import numpy as np

def optimize_threshold(model_path, data_path, output_dir):
    """Find optimal classification threshold."""
    
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION FOR 91.76% → 95%+ F1")
    print("=" * 70)
    
    # Load model and data
    print("\n📂 Loading model and data...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # Get model architecture from checkpoint
    import sys
    sys.path.append('scripts')
    from train_gat_simple_scaled import SimpleGATNode
    
    num_features = data.x.size(1)
    hidden_dim = 256
    num_classes = 2
    num_heads = 8
    num_layers = 3
    dropout = 0.3
    
    model = SimpleGATNode(
        in_channels=num_features,
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Model loaded (epoch {checkpoint['epoch']})")
    print(f"  ✓ Data loaded ({data.num_nodes} nodes, {data.num_edges} edges)")
    
    # Create data splits if they don't exist
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
        print("\n  ⚠️  Data splits not found. Creating new splits (80/10/10)...")
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(num_nodes * 0.8)
        val_size = int(num_nodes * 0.1)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        print(f"  ✓ Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Get predictions on validation set
    print("\n🔮 Generating predictions on validation set...")
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = F.softmax(logits, dim=1)
        probs_positive = probabilities[:, 1].numpy()  # Probability of "real" news
    
    val_mask = data.val_mask.numpy()
    y_true = data.y.numpy()[val_mask]
    y_probs = probs_positive[val_mask]
    
    print(f"  ✓ Validation set size: {val_mask.sum()} samples")
    print(f"  ✓ Positive class (real news): {(y_true == 1).sum()} samples")
    print(f"  ✓ Negative class (fake news): {(y_true == 0).sum()} samples")
    
    # Test default threshold (0.5)
    print("\n📊 Default Threshold (0.5) Performance:")
    y_pred_default = (y_probs >= 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default, average='weighted')
    precision_default = precision_score(y_true, y_pred_default, average='weighted')
    recall_default = recall_score(y_true, y_pred_default, average='weighted')
    
    print(f"  Precision: {precision_default:.4f}")
    print(f"  Recall:    {recall_default:.4f}")
    print(f"  F1 Score:  {f1_default:.4f}")
    
    # Find optimal threshold
    print("\n🎯 Finding optimal threshold...")
    thresholds = np.arange(0.1, 0.95, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precision_scores[optimal_idx]
    optimal_recall = recall_scores[optimal_idx]
    
    print(f"\n✅ OPTIMAL THRESHOLD FOUND: {optimal_threshold:.3f}")
    print(f"  Precision: {optimal_precision:.4f} ({optimal_precision*100:.2f}%)")
    print(f"  Recall:    {optimal_recall:.4f} ({optimal_recall*100:.2f}%)")
    print(f"  F1 Score:  {optimal_f1:.4f} ({optimal_f1*100:.2f}%)")
    
    improvement = optimal_f1 - f1_default
    print(f"\n📈 IMPROVEMENT: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")
    
    if improvement > 0:
        print(f"  🎉 Success! We gained {improvement*100:.2f} points just by tuning threshold!")
    else:
        print(f"  ℹ️  No improvement from threshold tuning. Model already uses optimal threshold.")
    
    # Test on test set with optimal threshold
    print("\n🧪 Testing optimal threshold on TEST set...")
    test_mask = data.test_mask.numpy()
    y_true_test = data.y.numpy()[test_mask]
    y_probs_test = probs_positive[test_mask]
    
    y_pred_test_default = (y_probs_test >= 0.5).astype(int)
    y_pred_test_optimal = (y_probs_test >= optimal_threshold).astype(int)
    
    f1_test_default = f1_score(y_true_test, y_pred_test_default, average='weighted')
    f1_test_optimal = f1_score(y_true_test, y_pred_test_optimal, average='weighted')
    
    precision_test_optimal = precision_score(y_true_test, y_pred_test_optimal, average='weighted')
    recall_test_optimal = recall_score(y_true_test, y_pred_test_optimal, average='weighted')
    
    print(f"  Default threshold (0.5):  F1 = {f1_test_default:.4f}")
    print(f"  Optimal threshold ({optimal_threshold:.3f}): F1 = {f1_test_optimal:.4f}")
    print(f"  Precision: {precision_test_optimal:.4f}")
    print(f"  Recall:    {recall_test_optimal:.4f}")
    print(f"  Improvement: {(f1_test_optimal - f1_test_default)*100:+.2f} points")
    
    # Create visualizations
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Creating visualizations...")
    
    # Plot 1: F1 score vs threshold
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Default (0.5)')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Classification Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Precision and Recall vs threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, precision_scores, 'g-', linewidth=2, label='Precision')
    plt.plot(thresholds, recall_scores, 'b-', linewidth=2, label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', linewidth=2, label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision-Recall-F1 Tradeoff', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'threshold_optimization.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: threshold_optimization.png")
    
    # Plot 3: ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: roc_curve.png")
    
    # Plot 4: Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: precision_recall_curve.png")
    
    # Save results
    results = {
        'optimal_threshold': float(optimal_threshold),
        'default_threshold': 0.5,
        'validation': {
            'default': {
                'f1': float(f1_default),
                'precision': float(precision_default),
                'recall': float(recall_default)
            },
            'optimal': {
                'f1': float(optimal_f1),
                'precision': float(optimal_precision),
                'recall': float(optimal_recall)
            }
        },
        'test': {
            'default': {
                'f1': float(f1_test_default)
            },
            'optimal': {
                'f1': float(f1_test_optimal),
                'precision': float(precision_test_optimal),
                'recall': float(recall_test_optimal)
            }
        },
        'improvement_val': float(improvement),
        'improvement_test': float(f1_test_optimal - f1_test_default),
        'roc_auc': float(roc_auc)
    }
    
    with open(output_path / 'threshold_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved: threshold_optimization_results.json")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline F1 (threshold=0.5):         {f1_test_default:.4f} ({f1_test_default*100:.2f}%)")
    print(f"Optimized F1 (threshold={optimal_threshold:.3f}):    {f1_test_optimal:.4f} ({f1_test_optimal*100:.2f}%)")
    print(f"Gain from threshold optimization:    {(f1_test_optimal - f1_test_default)*100:+.2f} points")
    print("\nProgress toward 95% F1 target:")
    progress = ((f1_test_optimal - 0.9176) / (0.95 - 0.9176)) * 100
    print(f"  Current: {f1_test_optimal*100:.2f}%")
    print(f"  Target:  95.00%")
    print(f"  Progress: {progress:.1f}% of the way there")
    print("=" * 70)
    
    return optimal_threshold, results

if __name__ == "__main__":
    model_path = "experiments/models_fullscale/gat_model_best_scaled.pt"
    data_path = "data/graphs_full/graph_data_enriched.pt"
    output_dir = "experiments/threshold_optimization"
    
    optimal_threshold, results = optimize_threshold(model_path, data_path, output_dir)
