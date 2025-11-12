"""
Compare results from different experiments
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_results(path: str) -> Dict:
    """Load results JSON"""
    with open(path) as f:
        return json.load(f)


def compare_metrics(results_list: List[Dict], labels: List[str]) -> pd.DataFrame:
    """Create comparison DataFrame"""
    
    rows = []
    
    for results, label in zip(results_list, labels):
        row = {'Experiment': label}
        
        # Test metrics
        test = results.get('test_metrics', {})
        row['Test F1'] = test.get('f1', 0)
        row['Test Acc'] = test.get('accuracy', 0)
        row['Test AUC'] = test.get('auc', 0)
        row['Test Prec'] = test.get('precision', 0)
        row['Test Rec'] = test.get('recall', 0)
        
        # Val metrics
        val = results.get('val_metrics', {})
        row['Val F1'] = val.get('f1', 0)
        
        # Model info
        row['Params'] = results.get('model_params', 0)
        row['Epochs'] = results.get('best_epoch', 0)
        row['Time (min)'] = results.get('training_time_seconds', 0) / 60
        
        # Hyperparameters
        hyper = results.get('hyperparameters', {})
        row['Hidden'] = hyper.get('hidden_dim', '-')
        row['Layers'] = hyper.get('num_layers', '-')
        row['Heads'] = hyper.get('num_heads', '-')
        row['Dropout'] = hyper.get('dropout', '-')
        row['LR'] = hyper.get('lr', '-')
        
        # Special features
        row['Split'] = hyper.get('split_method', 'random')
        row['Loss'] = hyper.get('loss', 'ce')
        row['Tuned Thresh'] = hyper.get('tune_threshold', False)
        row['Calibrated'] = results.get('temperature', 1.0) != 1.0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_comparison(df: pd.DataFrame):
    """Pretty print comparison"""
    
    print("\n" + "="*120)
    print("📊 EXPERIMENT COMPARISON")
    print("="*120)
    
    # Metrics comparison
    print("\n🎯 Test Metrics:")
    metrics_cols = ['Experiment', 'Test F1', 'Test Acc', 'Test AUC', 'Test Prec', 'Test Rec']
    metrics_df = df[metrics_cols].copy()
    
    # Highlight best
    for col in ['Test F1', 'Test Acc', 'Test AUC', 'Test Prec', 'Test Rec']:
        if col in metrics_df.columns:
            max_val = metrics_df[col].max()
            metrics_df[col] = metrics_df[col].apply(
                lambda x: f"**{x:.4f}**" if x == max_val else f"{x:.4f}"
            )
    
    print(metrics_df.to_string(index=False))
    
    # Model comparison
    print("\n🧠 Model Configuration:")
    config_cols = ['Experiment', 'Hidden', 'Layers', 'Heads', 'Dropout', 'LR', 'Params', 'Epochs']
    config_df = df[config_cols].copy()
    print(config_df.to_string(index=False))
    
    # Features comparison
    print("\n⚙️  Training Features:")
    features_cols = ['Experiment', 'Split', 'Loss', 'Tuned Thresh', 'Calibrated', 'Time (min)']
    features_df = df[features_cols].copy()
    features_df['Time (min)'] = features_df['Time (min)'].apply(lambda x: f"{x:.1f}")
    print(features_df.to_string(index=False))
    
    # Improvements
    if len(df) > 1:
        print("\n📈 Improvements vs Baseline:")
        baseline_f1 = df.iloc[0]['Test F1']
        
        for idx, row in df.iterrows():
            if idx == 0:
                continue
            
            f1_diff = row['Test F1'] - baseline_f1
            f1_pct = (f1_diff / baseline_f1) * 100
            
            arrow = "🔼" if f1_diff > 0 else "🔽"
            print(f"  {arrow} {row['Experiment']}: "
                  f"F1 {row['Test F1']:.4f} "
                  f"({f1_diff:+.4f}, {f1_pct:+.1f}%)")
    
    print("\n" + "="*120)


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('results_files', nargs='+', 
                       help='Paths to results.json files')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Custom labels for experiments')
    parser.add_argument('--output', type=str, default=None,
                       help='Save comparison to CSV')
    
    args = parser.parse_args()
    
    # Load results
    results_list = []
    labels = []
    
    for i, path in enumerate(args.results_files):
        try:
            results = load_results(path)
            results_list.append(results)
            
            # Generate label
            if args.labels and i < len(args.labels):
                label = args.labels[i]
            else:
                # Use parent directory name
                label = Path(path).parent.name
            
            labels.append(label)
            
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
            continue
    
    if not results_list:
        print("❌ No valid results files found")
        return
    
    # Create comparison
    df = compare_metrics(results_list, labels)
    
    # Print
    print_comparison(df)
    
    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n💾 Saved comparison to: {args.output}")
    
    # Print best config
    best_idx = df['Test F1'].astype(str).str.replace('*', '').astype(float).idxmax()
    best_exp = df.iloc[best_idx]
    
    print(f"\n🏆 BEST MODEL: {best_exp['Experiment']}")
    print(f"  • Test F1: {best_exp['Test F1']}")
    print(f"  • Test AUC: {best_exp['Test AUC']}")
    print(f"  • Configuration: {best_exp['Layers']} layers, "
          f"{best_exp['Hidden']} hidden, {best_exp['Heads']} heads")
    print(f"  • Training: {best_exp['Loss']} loss, {best_exp['Split']} split")


if __name__ == "__main__":
    main()
