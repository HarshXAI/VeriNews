"""
Analyze HPO results and identify top configurations for ensembling
"""

import json
from pathlib import Path

# Load HPO results
hpo_path = Path("experiments/hpo_memory_efficient/hpo_results.json")
with open(hpo_path, 'r') as f:
    hpo_results = json.load(f)

print("="*80)
print("HPO ANALYSIS - TOP PERFORMING TRIALS")
print("="*80)

# Sort trials by validation F1
completed_trials = [t for t in hpo_results['all_trials'] if t['state'] == '1']
completed_trials.sort(key=lambda x: x['value'], reverse=True)

print(f"\nTotal trials: {len(hpo_results['all_trials'])}")
print(f"Completed trials: {len(completed_trials)}")
print(f"Pruned trials: {len([t for t in hpo_results['all_trials'] if t['state'] == '2'])}")

print(f"\n{'='*80}")
print("TOP 10 CONFIGURATIONS")
print(f"{'='*80}\n")

for i, trial in enumerate(completed_trials[:10], 1):
    params = trial['params']
    val_f1 = trial['value']
    
    print(f"{i}. Trial {trial['number']} - Val F1: {val_f1:.4f} ({val_f1*100:.2f}%)")
    print(f"   hidden={params['hidden_dim']}, heads={params['num_heads']}, "
          f"layers={params['num_layers']}, dropout={params['dropout']:.2f}")
    print(f"   lr={params['lr']:.4f}, wd={params['weight_decay']:.6f}, gatv2={params['use_gatv2']}")
    print()

# Analyze patterns
print(f"{'='*80}")
print("PATTERN ANALYSIS")
print(f"{'='*80}\n")

top_5 = completed_trials[:5]

# Hidden dim distribution
hidden_dims = [t['params']['hidden_dim'] for t in top_5]
print(f"Hidden dims in top 5: {hidden_dims}")
print(f"  Most common: {max(set(hidden_dims), key=hidden_dims.count)}")

# Num heads distribution
num_heads = [t['params']['num_heads'] for t in top_5]
print(f"\nNum heads in top 5: {num_heads}")
print(f"  Most common: {max(set(num_heads), key=num_heads.count)}")

# Num layers distribution
num_layers = [t['params']['num_layers'] for t in top_5]
print(f"\nNum layers in top 5: {num_layers}")
print(f"  Most common: {max(set(num_layers), key=num_layers.count)}")

# Dropout distribution
dropouts = [t['params']['dropout'] for t in top_5]
print(f"\nDropout in top 5: {dropouts}")
print(f"  Average: {sum(dropouts)/len(dropouts):.2f}")

# GATv2 usage
gatv2_usage = [t['params']['use_gatv2'] for t in top_5]
print(f"\nGATv2 in top 5: {gatv2_usage}")
print(f"  GATv2 count: {sum(gatv2_usage)}, GAT count: {len(gatv2_usage) - sum(gatv2_usage)}")

print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}\n")

print("1. Best single model achieved: 88.56% F1")
print("2. Gap to 95% target: 6.44 points (very challenging!)")
print("3. Gap to realistic 90% target: 1.44 points (achievable)")
print("\nNext steps:")
print("  • Ensemble top-3 models → potential +0.5-1.0 points")
print("  • Feature engineering → potential +1-2 points")
print("  • Graph augmentation → potential +0.5-1.5 points")
print("  • Advanced architectures (GIN, GraphSAINT) → potential +1-2 points")
print("\nTo reach 90% F1: Ensemble should be sufficient")
print("To reach 95% F1: Need multiple advanced techniques combined")
