"""
Model evaluation script

Usage:
    python scripts/evaluate_model.py --checkpoint experiments/best_model.pt --data data/graphs
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json

from src.models import FakeNewsGAT
from src.evaluation import ModelEvaluator, AttentionAnalyzer
from src.visualization import GraphVisualizer, MetricsVisualizer
from src.utils import load_checkpoint, get_device, print_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GAT model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/graphs/test_data.pt",
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--analyze-attention",
        action="store_true",
        help="Perform attention analysis"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("\nPlease train a model first:")
        print("  python scripts/train_model.py")
        sys.exit(1)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    
    # Create model (adjust parameters based on your training)
    model = FakeNewsGAT(
        in_channels=768,  # BERT embedding dimension
        hidden_channels=128,
        out_channels=2,
        num_layers=3,
        num_heads=8,
        dropout=0.3
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Load test data
    if not os.path.exists(args.data):
        print(f"\n⚠️  Warning: Test data not found at {args.data}")
        print("Skipping evaluation...")
        return
    
    print(f"\nLoading test data from {args.data}...")
    test_data = torch.load(args.data)
    
    # Create test loader
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator(model, device)
    metrics, y_pred, y_true = evaluator.evaluate(test_loader)
    
    # Print metrics
    print_metrics(metrics, "Test Set Metrics")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(args.output, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Visualize metrics
    print("\nGenerating visualizations...")
    visualizer = MetricsVisualizer()
    
    # Plot metrics comparison
    visualizer.plot_metrics_comparison(
        metrics,
        save_path=os.path.join(args.output, "metrics_comparison.png")
    )
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    visualizer.plot_confusion_matrix(
        cm,
        class_names=['Fake', 'Real'],
        save_path=os.path.join(args.output, "confusion_matrix.png")
    )
    
    print(f"\nVisualizations saved to {args.output}/")
    
    # Attention analysis (if requested)
    if args.analyze_attention:
        print("\n" + "="*60)
        print("ATTENTION ANALYSIS")
        print("="*60)
        
        analyzer = AttentionAnalyzer(model, device)
        
        # Analyze first batch
        batch = next(iter(test_loader))
        batch = batch.to(device)
        
        attention_weights = analyzer.get_attention_weights(
            batch.x,
            batch.edge_index,
            batch.batch
        )
        
        # Find influential users
        influential = analyzer.identify_influential_users(
            attention_weights,
            top_k=20
        )
        
        print("\nTop 20 Influential Nodes:")
        for i, (node_id, score) in enumerate(influential, 1):
            print(f"  {i:2d}. Node {node_id:4d}: {score:.4f}")
        
        # Save attention analysis
        attention_results = {
            'top_influential': [
                {'node_id': int(node_id), 'score': float(score)}
                for node_id, score in influential
            ]
        }
        
        attention_path = os.path.join(args.output, "attention_analysis.json")
        with open(attention_path, 'w') as f:
            json.dump(attention_results, f, indent=2)
        
        print(f"\nAttention analysis saved to {attention_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
