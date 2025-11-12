"""
Compare baseline vs improved GAT results
"""

import json
import sys

# Load results
with open('experiments/baseline_100epochs/results.json', 'r') as f:
    baseline = json.load(f)

with open('experiments/improved_100epochs/results.json', 'r') as f:
    improved = json.load(f)

print("=" * 70)
print("📊 BASELINE vs IMPROVED GAT COMPARISON")
print("=" * 70)

print("\n🔧 Model Configurations:")
print(f"  Baseline:  {baseline['num_parameters']:,} parameters")
print(f"  Improved:  {improved['num_parameters']:,} parameters (+{improved['num_parameters']-baseline['num_parameters']:,})")

print("\n📈 Test Performance:")
print(f"\n  {'Metric':<15} {'Baseline':>12} {'Improved':>12} {'Δ':>12}")
print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

metrics = [
    ('Accuracy', 'test_accuracy'),
    ('Precision', 'test_precision'),
    ('Recall', 'test_recall'),
    ('F1 Score', 'test_f1'),
    ('F1 (tuned)', 'test_f1_tuned'),
    ('AUC', 'test_auc'),
]

for name, key in metrics:
    b_val = baseline.get(key, 0)
    i_val = improved.get(key, 0)
    delta = i_val - b_val
    
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    color = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
    
    print(f"  {name:<15} {b_val:>12.4f} {i_val:>12.4f} {delta_str:>11} {color}")

print("\n🎯 Improvements:")
print(f"  • GATv2 with residual connections")
print(f"  • Focal Loss (α={improved['config']['focal_alpha']}, γ={improved['config']['focal_gamma']})")
print(f"  • DropEdge regularization (p={improved['config']['drop_edge']})")
print(f"  • Temperature scaling for calibration")
print(f"  • Threshold tuning (optimal={improved['threshold']:.2f})")

print("\n✅ F1 Score Improvement:")
baseline_f1 = baseline['test_f1']
improved_f1 = improved['test_f1_tuned']
improvement = improved_f1 - baseline_f1
improvement_pct = (improvement / baseline_f1) * 100

print(f"  {baseline_f1:.4f} → {improved_f1:.4f} (+{improvement:.4f}, +{improvement_pct:.2f}%)")

print("\n" + "=" * 70)
