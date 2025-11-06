"""
Week 2: Model ensembling for improved performance
Train multiple models and combine predictions
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def load_model(checkpoint_path, in_channels, device):
    """Load a trained model"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Infer architecture from checkpoint or use defaults
    model = SimpleGATNode(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=2,
        num_heads=8,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


@torch.no_grad()
def get_predictions(model, data, device):
    """Get model predictions"""
    out = model(data.x.to(device), data.edge_index.to(device))
    probs = F.softmax(out, dim=1).cpu().numpy()
    return probs


def ensemble_predict(model_paths, data, device, method='average'):
    """
    Ensemble predictions from multiple models
    
    Args:
        model_paths: List of checkpoint paths
        data: PyG Data object
        device: Compute device
        method: 'average', 'weighted', or 'voting'
    """
    print(f"\n🎯 Ensemble prediction with {len(model_paths)} models...")
    print(f"  Method: {method}")
    
    all_probs = []
    
    for i, path in enumerate(tqdm(model_paths, desc="  Loading models")):
        try:
            model = load_model(path, data.x.shape[1], device)
            probs = get_predictions(model, data, device)
            all_probs.append(probs)
        except Exception as e:
            print(f"    ⚠️  Failed to load {path}: {e}")
    
    if not all_probs:
        raise ValueError("No models loaded successfully")
    
    # Combine predictions
    if method == 'average':
        # Simple average
        ensemble_probs = np.mean(all_probs, axis=0)
    
    elif method == 'weighted':
        # Weight by model confidence (higher confidence = more weight)
        weights = []
        for probs in all_probs:
            # Use max probability as confidence
            confidence = np.max(probs, axis=1).mean()
            weights.append(confidence)
        
        weights = np.array(weights) / sum(weights)
        print(f"  Model weights: {weights}")
        
        ensemble_probs = np.average(all_probs, axis=0, weights=weights)
    
    elif method == 'voting':
        # Hard voting
        preds = [np.argmax(probs, axis=1) for probs in all_probs]
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.array(preds)
        )
        # Convert back to probabilities (one-hot)
        ensemble_probs = np.zeros((len(ensemble_pred), 2))
        ensemble_probs[np.arange(len(ensemble_pred)), ensemble_pred] = 1.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble_probs


def evaluate_ensemble(ensemble_probs, data, mask):
    """Evaluate ensemble predictions"""
    pred = np.argmax(ensemble_probs[mask], axis=1)
    y_true = data.y[mask].cpu().numpy()
    
    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    
    try:
        probs_positive = ensemble_probs[mask][:, 1]
        auc = roc_auc_score(y_true, probs_positive)
    except:
        auc = 0.0
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/graphs_full/graph_data_enriched.pt")
    parser.add_argument("--model-dir", type=str, default="experiments/models")
    parser.add_argument("--model-pattern", type=str, default="gat_model_best*.pt")
    parser.add_argument("--method", type=str, default="average", choices=['average', 'weighted', 'voting'])
    parser.add_argument("--output-path", type=str, default="experiments/ensemble_results.json")
    parser.add_argument("--device", type=str, default="mps")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎭 MODEL ENSEMBLE")
    print("="*70)
    
    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data...")
    data = torch.load(args.data_path, weights_only=False)
    print(f"  ✓ Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    
    # Find model checkpoints
    from glob import glob
    model_pattern = os.path.join(args.model_dir, args.model_pattern)
    model_paths = glob(model_pattern)
    
    if not model_paths:
        print(f"  ⚠️  No models found matching: {model_pattern}")
        print(f"  Tip: Train multiple models first or specify correct pattern")
        return
    
    print(f"\n📦 Found {len(model_paths)} models:")
    for path in model_paths:
        print(f"  • {os.path.basename(path)}")
    
    # Create test mask if not exists
    if not hasattr(data, 'test_mask'):
        print("\n  Creating test split...")
        num_nodes = data.x.shape[0]
        indices = torch.randperm(num_nodes)
        test_size = int(num_nodes * 0.15)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[indices[:test_size]] = True
        data.test_mask = test_mask
    
    # Get ensemble predictions
    ensemble_probs = ensemble_predict(model_paths, data, device, method=args.method)
    
    # Evaluate
    print("\n📊 Evaluating ensemble...")
    metrics = evaluate_ensemble(ensemble_probs, data, data.test_mask)
    
    print("\n" + "="*70)
    print("✅ ENSEMBLE RESULTS")
    print("="*70)
    print(f"\nTest Set Performance:")
    print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  • Precision: {metrics['precision']:.4f}")
    print(f"  • Recall:    {metrics['recall']:.4f}")
    print(f"  • F1-Score:  {metrics['f1']:.4f}")
    print(f"  • AUC:       {metrics['auc']:.4f}")
    
    # Compare with individual models
    print(f"\nIndividual Model Performance:")
    for path in model_paths:
        try:
            model = load_model(path, data.x.shape[1], device)
            probs = get_predictions(model, data, device)
            ind_metrics = evaluate_ensemble(probs, data, data.test_mask)
            print(f"  • {os.path.basename(path)}: F1={ind_metrics['f1']:.4f}")
        except:
            pass
    
    # Save results
    results = {
        'method': args.method,
        'num_models': len(model_paths),
        'ensemble_metrics': metrics,
        'model_paths': model_paths
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
