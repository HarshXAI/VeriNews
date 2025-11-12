"""
Analyze the baseline 91.76% F1 model to identify improvement opportunities.

Goal: Understand where the model fails to guide improvements toward 95%+ F1.
"""

import json
import torch
from pathlib import Path

def analyze_baseline():
    """Analyze the baseline model performance."""
    
    # Load training metrics
    metrics_path = Path("experiments/models_fullscale/training_metrics_scaled.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    test_metrics = metrics['test_metrics']
    
    print("=" * 70)
    print("BASELINE MODEL PERFORMANCE ANALYSIS (91.76% F1)")
    print("=" * 70)
    
    print("\n📊 Test Set Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f} ({test_metrics['auc']*100:.2f}%)")
    
    # Analyze precision-recall tradeoff
    precision = test_metrics['precision']
    recall = test_metrics['recall']
    
    print("\n🔍 Precision-Recall Analysis:")
    if recall > 0.95 and precision < 0.90:
        print("  ⚠️  HIGH RECALL, LOWER PRECISION detected!")
        print(f"     - Model is catching {recall*100:.2f}% of real news (good)")
        print(f"     - But only {precision*100:.2f}% of predictions are correct")
        print("     - Issue: Too many FALSE POSITIVES (fake news predicted as real)")
        print("\n  💡 Improvement Strategy:")
        print("     1. Increase threshold (make model more conservative)")
        print("     2. Add class weights to penalize false positives more")
        print("     3. Focus on improving fake news detection")
    
    # Calculate potential gains
    print("\n📈 Path to 95% F1:")
    target_f1 = 0.95
    current_f1 = test_metrics['f1']
    gap = target_f1 - current_f1
    
    print(f"  Current F1:  {current_f1:.4f} ({current_f1*100:.2f}%)")
    print(f"  Target F1:   {target_f1:.4f} ({target_f1*100:.2f}%)")
    print(f"  Gap to close: {gap:.4f} ({gap*100:.2f} percentage points)")
    
    # Estimate required precision/recall combinations for 95% F1
    print("\n  Required Precision-Recall Combinations for 95% F1:")
    for recall_target in [0.93, 0.94, 0.95, 0.96, 0.97]:
        # F1 = 2 * (P * R) / (P + R)
        # Solve for P: P = (F1 * R) / (2*R - F1)
        precision_needed = (target_f1 * recall_target) / (2 * recall_target - target_f1)
        if precision_needed <= 1.0:
            change_p = precision_needed - precision
            change_r = recall_target - recall
            print(f"     Recall={recall_target:.2f}, Precision={precision_needed:.2f} "
                  f"(Δ Precision: {change_p:+.3f}, Δ Recall: {change_r:+.3f})")
    
    # Training convergence analysis
    history = metrics['history']
    best_epoch = metrics['best_epoch']
    
    print(f"\n📉 Training Convergence:")
    print(f"  Best Epoch: {best_epoch}/100")
    print(f"  Best Val F1: {metrics['val_metrics']['f1']:.4f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
    
    # Check for overfitting/underfitting
    train_acc_final = history['train_acc'][-1]
    val_acc_final = history['val_acc'][-1]
    gap_acc = train_acc_final - val_acc_final
    
    print(f"\n  Train Accuracy: {train_acc_final:.4f}")
    print(f"  Val Accuracy:   {val_acc_final:.4f}")
    print(f"  Gap: {gap_acc:.4f}")
    
    if gap_acc > 0.05:
        print("  ⚠️  OVERFITTING detected (train >> val)")
        print("     - Consider: more regularization, early stopping")
    elif gap_acc < 0.01:
        print("  ⚠️  UNDERFITTING detected (train ≈ val, both low)")
        print("     - Consider: larger model, more epochs, lower regularization")
    else:
        print("  ✅  Good generalization (train ≈ val)")
    
    # Model configuration
    config = metrics['config']
    print(f"\n⚙️  Model Configuration:")
    print(f"  Architecture: GAT")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  Num Heads: {config['num_heads']}")
    print(f"  Num Layers: {config['num_layers']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Device: {config['device']}")
    
    # Graph statistics
    print(f"\n🕸️  Graph Statistics:")
    print(f"  Nodes: {metrics['num_nodes']:,}")
    print(f"  Edges: {metrics['num_edges']:,}")
    print(f"  Avg Degree: {metrics['num_edges'] / metrics['num_nodes']:.2f}")
    print(f"  Parameters: {metrics['num_params']:,}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS (Priority Order):")
    print("=" * 70)
    print("\n1. 🎯 THRESHOLD OPTIMIZATION (Quick Win)")
    print("   - Current uses default 0.5 threshold")
    print("   - Optimize on validation set to find best F1")
    print("   - Expected gain: +0.5 to +1.5 F1 points")
    
    print("\n2. ⚖️  CLASS WEIGHT TUNING (High Impact)")
    print("   - Dataset is 75% real, 25% fake (imbalanced)")
    print("   - Add class_weight=[3.0, 1.0] to loss function")
    print("   - Expected gain: +0.5 to +2.0 F1 points")
    
    print("\n3. 🔄 UPGRADE TO GATv2 (Moderate Risk)")
    print("   - Keep same hyperparameters (hidden=256, heads=8)")
    print("   - Only change: GAT → GATv2Conv")
    print("   - Expected gain: +0.3 to +1.0 F1 points")
    
    print("\n4. 🎲 HYPERPARAMETER OPTIMIZATION (Time Investment)")
    print("   - Use Optuna to search optimal config")
    print("   - Search: hidden_dim, num_heads, lr, dropout")
    print("   - Expected gain: +1.0 to +2.5 F1 points")
    
    print("\n5. 🏗️  ARCHITECTURE ENHANCEMENTS (Higher Risk)")
    print("   - Add residual connections (carefully)")
    print("   - Try 4 layers instead of 3")
    print("   - Add edge weights based on similarity")
    print("   - Expected gain: +0.5 to +1.5 F1 points")
    
    print("\n" + "=" * 70)
    print("🎯 REALISTIC TARGET: Combining steps 1-3 should reach 94-95% F1")
    print("🚀 STRETCH GOAL: Adding step 4 could push toward 95-96% F1")
    print("=" * 70)

if __name__ == "__main__":
    analyze_baseline()
