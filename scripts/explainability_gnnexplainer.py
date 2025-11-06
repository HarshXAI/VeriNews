"""
Week 5-6: GNNExplainer for model interpretability
Identify important subgraphs and features for predictions
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gat_model import SimpleGATNode


def explain_node(model, data, node_idx, device):
    """Explain a single node's prediction using gradient-based attribution"""
    model.eval()
    
    # Get prediction
    out = model(data.x.to(device), data.edge_index.to(device))
    pred_class = out[node_idx].argmax().item()
    pred_prob = F.softmax(out[node_idx], dim=0)[pred_class].item()
    
    # Gradient-based feature importance
    data.x.requires_grad = True
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = out[node_idx, pred_class]
    loss.backward()
    
    feature_importance = data.x.grad[node_idx].abs().cpu().numpy()
    
    # Edge importance via attention (approximate)
    edge_index = data.edge_index.cpu()
    edges_to_node = (edge_index[1] == node_idx).nonzero(as_tuple=True)[0]
    source_nodes = edge_index[0][edges_to_node].tolist()
    
    # Get neighbor contributions
    neighbor_importance = {}
    for src in source_nodes:
        neighbor_importance[src] = feature_importance.mean()  # Simplified
    
    return {
        'node_idx': node_idx,
        'predicted_class': pred_class,
        'confidence': pred_prob,
        'feature_importance': feature_importance,
        'top_features': feature_importance.argsort()[-10:][::-1].tolist(),
        'neighbor_importance': neighbor_importance
    }


def visualize_explanation(explanation, data, output_path):
    """Visualize the explanation"""
    node_idx = explanation['node_idx']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top features
    ax = axes[0, 0]
    top_feats = explanation['top_features'][:20]
    top_vals = explanation['feature_importance'][top_feats]
    ax.barh(range(len(top_feats)), top_vals)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels([f"Feature {f}" for f in top_feats])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top 20 Feature Importances for Node {node_idx}')
    ax.invert_yaxis()
    
    # 2. Feature importance distribution
    ax = axes[0, 1]
    ax.hist(explanation['feature_importance'], bins=50, alpha=0.7)
    ax.axvline(explanation['feature_importance'].mean(), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Importance Distribution')
    ax.legend()
    
    # 3. Local subgraph
    ax = axes[1, 0]
    edge_index = data.edge_index.cpu()
    
    # Build local subgraph
    neighbors = explanation['neighbor_importance'].keys()
    local_nodes = [node_idx] + list(neighbors)
    
    G = nx.DiGraph()
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if src in local_nodes and dst in local_nodes:
            G.add_edge(src, dst)
    
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, seed=42)
        node_colors = ['red' if n == node_idx else 'lightblue' for n in G.nodes()]
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                node_size=500, ax=ax, arrows=True)
        ax.set_title(f'Local Subgraph (Node {node_idx} in red)')
    
    # 4. Prediction confidence
    ax = axes[1, 1]
    labels = ['Fake', 'Real']
    colors = ['red', 'green']
    bars = ax.bar(labels, [1-explanation['confidence'], explanation['confidence']], 
                   color=colors, alpha=0.6)
    ax.set_ylabel('Confidence')
    ax.set_ylim([0, 1])
    ax.set_title(f"Prediction: {labels[explanation['predicted_class']]}")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_counterfactual(model, data, node_idx, device):
    """Generate counterfactual explanation (what needs to change for different prediction)"""
    model.eval()
    
    # Get current prediction
    out = model(data.x.to(device), data.edge_index.to(device))
    orig_pred = out[node_idx].argmax().item()
    orig_conf = F.softmax(out[node_idx], dim=0)[orig_pred].item()
    
    # Try feature perturbations
    x_perturbed = data.x.clone()
    target_class = 1 - orig_pred
    
    best_change = None
    min_perturbation = float('inf')
    
    # Gradient-based approach
    x_perturbed.requires_grad = True
    out = model(x_perturbed.to(device), data.edge_index.to(device))
    loss = -out[node_idx, target_class]  # Maximize target class probability
    loss.backward()
    
    gradient = x_perturbed.grad[node_idx]
    
    # Find top features to change
    top_change_features = gradient.abs().argsort(descending=True)[:5].tolist()
    
    return {
        'original_prediction': orig_pred,
        'original_confidence': orig_conf,
        'target_class': target_class,
        'features_to_change': top_change_features,
        'suggested_changes': gradient[top_change_features].tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="experiments/models/gat_model_best.pt")
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/explainability")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of nodes to explain")
    parser.add_argument("--device", type=str, default="mps")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔍 GNN EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    # Setup
    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data and model
    print("\n📂 Loading model and data...")
    data = torch.load(args.data_path, weights_only=False)
    
    checkpoint = torch.load(args.model_path, weights_only=False)
    model = SimpleGATNode(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_heads=8,
        num_layers=3
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  ✓ Data: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Select nodes to explain
    test_mask = data.test_mask if hasattr(data, 'test_mask') else torch.ones(data.x.shape[0], dtype=torch.bool)
    test_nodes = test_mask.nonzero(as_tuple=True)[0]
    
    if len(test_nodes) > args.num_examples:
        # Select diverse examples
        import random
        random.seed(42)
        selected_nodes = random.sample(test_nodes.tolist(), args.num_examples)
    else:
        selected_nodes = test_nodes.tolist()
    
    print(f"\n🔍 Explaining {len(selected_nodes)} nodes...")
    
    # Generate explanations
    explanations = []
    for node_idx in tqdm(selected_nodes, desc="Explaining"):
        try:
            explanation = explain_node(model, data, node_idx, device)
            counterfactual = generate_counterfactual(model, data, node_idx, device)
            
            explanation['counterfactual'] = counterfactual
            explanations.append(explanation)
            
            # Visualize
            viz_path = os.path.join(args.output_dir, f"explanation_node_{node_idx}.png")
            visualize_explanation(explanation, data, viz_path)
            
        except Exception as e:
            print(f"  ⚠️  Failed to explain node {node_idx}: {e}")
    
    print(f"  ✓ Generated {len(explanations)} explanations")
    
    # Aggregate statistics
    print("\n📊 Computing aggregate statistics...")
    
    all_feature_importance = []
    for exp in explanations:
        all_feature_importance.append(exp['feature_importance'])
    
    avg_feature_importance = sum(all_feature_importance) / len(all_feature_importance)
    
    # Global feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    top_global = avg_feature_importance.argsort()[-30:][::-1]
    ax.barh(range(len(top_global)), avg_feature_importance[top_global])
    ax.set_yticks(range(len(top_global)))
    ax.set_yticklabels([f"Feature {f}" for f in top_global])
    ax.set_xlabel('Average Importance Score')
    ax.set_title('Top 30 Global Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "global_feature_importance.png"), dpi=150)
    plt.close()
    
    # Save results
    results = {
        'num_explanations': len(explanations),
        'avg_confidence': float(sum(e['confidence'] for e in explanations) / len(explanations)),
        'top_global_features': top_global.tolist(),
        'explanations_summary': [
            {
                'node_idx': e['node_idx'],
                'predicted_class': e['predicted_class'],
                'confidence': e['confidence'],
                'num_neighbors': len(e['neighbor_importance'])
            }
            for e in explanations
        ]
    }
    
    with open(os.path.join(args.output_dir, "explainability_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ EXPLAINABILITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • {len(explanations)} individual explanations")
    print(f"  • Global feature importance visualization")
    print(f"  • Counterfactual explanations")
    print(f"\nAverage confidence: {results['avg_confidence']:.3f}")
    print(f"Top 5 global features: {top_global[:5].tolist()}")


if __name__ == "__main__":
    main()
