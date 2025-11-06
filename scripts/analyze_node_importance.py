"""
Analyze node importance and generate prediction examples
"""

import argparse
import os
import sys
from pathlib import Path
import json
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import degree
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device


class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index, return_attention_weights=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention_weights:
            x, (edge_index1, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index)
            
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (edge_index2, alpha2) = self.conv2(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index)
            
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (edge_index3, alpha3) = self.conv3(x, edge_index, return_attention_weights=True)
            return F.log_softmax(x, dim=1), [(edge_index1, alpha1), (edge_index2, alpha2), (edge_index3, alpha3)]
        else:
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)


def compute_node_importance(data, model, attention_weights):
    """Compute various node importance metrics"""
    
    # Get predictions and confidence
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)
        pred = out.argmax(dim=1)
        confidence = probs.max(dim=1)[0]
    
    # Compute degree centrality
    edge_index = data.edge_index.cpu()
    in_degree = degree(edge_index[1], num_nodes=data.num_nodes).cpu().numpy()
    out_degree = degree(edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    total_degree = in_degree + out_degree
    
    # Compute attention received (sum of incoming attention)
    attention_received = np.zeros(data.num_nodes)
    attention_given = np.zeros(data.num_nodes)
    
    for layer_idx in range(len(attention_weights)):
        edge_idx, alpha = attention_weights[layer_idx]
        edge_idx_np = edge_idx.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        
        # Average across heads if needed
        if len(alpha_np.shape) > 1:
            alpha_np = alpha_np.mean(axis=1)
        
        # Sum attention for each node
        for i in range(edge_idx_np.shape[1]):
            src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
            attention_received[dst] += alpha_np[i]
            attention_given[src] += alpha_np[i]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'node_id': range(data.num_nodes),
        'true_label': data.y.cpu().numpy(),
        'predicted_label': pred.cpu().numpy(),
        'confidence': confidence.cpu().numpy(),
        'prob_fake': probs[:, 1].cpu().numpy(),
        'prob_real': probs[:, 0].cpu().numpy(),
        'in_degree': in_degree,
        'out_degree': out_degree,
        'total_degree': total_degree,
        'attention_received': attention_received,
        'attention_given': attention_given,
        'correct': (pred.cpu().numpy() == data.y.cpu().numpy()).astype(int)
    })
    
    importance_df['label_name'] = importance_df['true_label'].map({0: 'Real', 1: 'Fake'})
    importance_df['prediction_name'] = importance_df['predicted_label'].map({0: 'Real', 1: 'Fake'})
    
    return importance_df


def plot_node_importance(importance_df, output_dir):
    """Plot node importance metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Degree distribution by label
    ax = axes[0, 0]
    for label in [0, 1]:
        label_name = 'Real' if label == 0 else 'Fake'
        data = importance_df[importance_df['true_label'] == label]['total_degree']
        ax.hist(data, bins=20, alpha=0.6, label=label_name, color='#4dabf7' if label == 0 else '#ff6b6b')
    ax.set_xlabel('Total Degree', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Node Degree Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Attention received vs degree
    ax = axes[0, 1]
    colors = importance_df['true_label'].map({0: '#4dabf7', 1: '#ff6b6b'})
    ax.scatter(importance_df['total_degree'], importance_df['attention_received'], 
              c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Total Degree', fontsize=11)
    ax.set_ylabel('Attention Received', fontsize=11)
    ax.set_title('Attention vs Degree', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence distribution
    ax = axes[0, 2]
    correct = importance_df[importance_df['correct'] == 1]['confidence']
    incorrect = importance_df[importance_df['correct'] == 0]['confidence']
    ax.hist(correct, bins=20, alpha=0.6, label='Correct', color='#6bcf7f')
    ax.hist(incorrect, bins=20, alpha=0.6, label='Incorrect', color='#ff6b6b')
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Attention received by label
    ax = axes[1, 0]
    data_to_plot = [
        importance_df[importance_df['true_label'] == 0]['attention_received'],
        importance_df[importance_df['true_label'] == 1]['attention_received']
    ]
    bp = ax.boxplot(data_to_plot, labels=['Real', 'Fake'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#4dabf7')
    bp['boxes'][1].set_facecolor('#ff6b6b')
    ax.set_ylabel('Attention Received', fontsize=11)
    ax.set_title('Attention by Node Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Prediction errors
    ax = axes[1, 1]
    error_types = importance_df[importance_df['correct'] == 0].groupby(['label_name', 'prediction_name']).size()
    if len(error_types) > 0:
        error_types.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4dabf7'])
        ax.set_xlabel('(True Label, Predicted Label)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Prediction Errors', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Top nodes by attention
    ax = axes[1, 2]
    top_nodes = importance_df.nlargest(10, 'attention_received')
    colors_top = top_nodes['true_label'].map({0: '#4dabf7', 1: '#ff6b6b'})
    ax.barh(range(len(top_nodes)), top_nodes['attention_received'], color=colors_top)
    ax.set_yticks(range(len(top_nodes)))
    ax.set_yticklabels([f"Node {nid}" for nid in top_nodes['node_id']])
    ax.set_xlabel('Attention Received', fontsize=11)
    ax.set_title('Top 10 Most Attended Nodes', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'node_importance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved node importance visualization")


def generate_prediction_examples(importance_df, data, output_dir, num_examples=20):
    """Generate detailed prediction examples"""
    
    # Load original data if available
    try:
        with open('data/graphs/propagation_graph.pkl', 'rb') as f:
            G = pickle.load(f)
        has_metadata = True
    except:
        has_metadata = False
        G = None
    
    examples = []
    
    # Get interesting cases
    # 1. High confidence correct predictions (5 fake, 5 real)
    correct_fake = importance_df[
        (importance_df['correct'] == 1) & 
        (importance_df['true_label'] == 1)
    ].nlargest(5, 'confidence')
    
    correct_real = importance_df[
        (importance_df['correct'] == 1) & 
        (importance_df['true_label'] == 0)
    ].nlargest(5, 'confidence')
    
    # 2. Incorrect predictions (all of them, up to 5 each type)
    incorrect_fake_as_real = importance_df[
        (importance_df['correct'] == 0) & 
        (importance_df['true_label'] == 1) &
        (importance_df['predicted_label'] == 0)
    ].nlargest(5, 'confidence')
    
    incorrect_real_as_fake = importance_df[
        (importance_df['correct'] == 0) & 
        (importance_df['true_label'] == 0) &
        (importance_df['predicted_label'] == 1)
    ].nlargest(5, 'confidence')
    
    # Combine
    selected_nodes = pd.concat([
        correct_fake, correct_real, 
        incorrect_fake_as_real, incorrect_real_as_fake
    ])
    
    for _, row in selected_nodes.iterrows():
        node_id = int(row['node_id'])
        
        # Get neighbors
        edge_index = data.edge_index.cpu().numpy()
        neighbors_out = edge_index[1, edge_index[0] == node_id]
        neighbors_in = edge_index[0, edge_index[1] == node_id]
        
        # Get neighbor labels
        if len(neighbors_out) > 0:
            out_labels = data.y[neighbors_out].cpu().numpy()
            out_fake_ratio = (out_labels == 1).mean()
        else:
            out_fake_ratio = 0
            
        if len(neighbors_in) > 0:
            in_labels = data.y[neighbors_in].cpu().numpy()
            in_fake_ratio = (in_labels == 1).mean()
        else:
            in_fake_ratio = 0
        
        example = {
            'node_id': node_id,
            'true_label': row['label_name'],
            'predicted_label': row['prediction_name'],
            'confidence': float(row['confidence']),
            'prob_fake': float(row['prob_fake']),
            'prob_real': float(row['prob_real']),
            'correct': bool(row['correct']),
            'in_degree': int(row['in_degree']),
            'out_degree': int(row['out_degree']),
            'attention_received': float(row['attention_received']),
            'attention_given': float(row['attention_given']),
            'out_neighbors_fake_ratio': float(out_fake_ratio),
            'in_neighbors_fake_ratio': float(in_fake_ratio),
        }
        
        # Add metadata if available
        if has_metadata and G is not None:
            node_data = G.nodes.get(node_id, {})
            example['title'] = node_data.get('title', 'N/A')[:100]
            example['text'] = node_data.get('text', 'N/A')[:200]
        
        examples.append(example)
    
    return examples


def create_prediction_report(examples, output_dir):
    """Create HTML report with prediction examples"""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>GAT Prediction Examples</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .example { 
            background: white; 
            margin: 20px 0; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .correct { border-left: 5px solid #6bcf7f; }
        .incorrect { border-left: 5px solid #ff6b6b; }
        .header { display: flex; justify-content: space-between; margin-bottom: 15px; }
        .label { padding: 5px 10px; border-radius: 4px; font-weight: bold; display: inline-block; }
        .fake { background: #ffe0e0; color: #d63031; }
        .real { background: #e0f2ff; color: #0984e3; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }
        .metric { background: #f8f9fa; padding: 10px; border-radius: 4px; text-align: center; }
        .metric-label { font-size: 12px; color: #666; }
        .metric-value { font-size: 18px; font-weight: bold; color: #333; }
        .text-content { background: #fafafa; padding: 15px; border-radius: 4px; margin-top: 15px; }
        .confidence-bar { height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #6bcf7f, #4dabf7); }
        .network-info { background: #fff8e1; padding: 10px; border-radius: 4px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🔍 GAT Model Prediction Examples</h1>
    <p style="text-align: center; color: #666;">Detailed analysis of model predictions with network context</p>
"""
    
    # Group by correctness
    correct_examples = [e for e in examples if e['correct']]
    incorrect_examples = [e for e in examples if not e['correct']]
    
    # Correct predictions
    html += "<h2>✅ Correct Predictions (High Confidence)</h2>"
    for ex in correct_examples:
        html += f"""
        <div class="example correct">
            <div class="header">
                <div>
                    <strong>Node {ex['node_id']}</strong>
                    <span class="label {ex['true_label'].lower()}">{ex['true_label']}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #666;">Confidence</div>
                    <div style="font-size: 20px; font-weight: bold; color: #6bcf7f;">{ex['confidence']:.1%}</div>
                </div>
            </div>
            
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {ex['confidence']*100}%;"></div>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Prob(Fake)</div>
                    <div class="metric-value">{ex['prob_fake']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Prob(Real)</div>
                    <div class="metric-value">{ex['prob_real']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Degree</div>
                    <div class="metric-value">{ex['in_degree'] + ex['out_degree']}</div>
                </div>
            </div>
            
            <div class="network-info">
                <strong>📊 Network Context:</strong><br>
                • Connections to this node: {ex['in_degree']} ({ex['in_neighbors_fake_ratio']:.0%} fake)<br>
                • Connections from this node: {ex['out_degree']} ({ex['out_neighbors_fake_ratio']:.0%} fake)<br>
                • Attention received: {ex['attention_received']:.2f} | Attention given: {ex['attention_given']:.2f}
            </div>
"""
        
        if 'title' in ex:
            html += f"""
            <div class="text-content">
                <strong>Title:</strong> {ex['title']}<br>
                <strong>Text:</strong> {ex['text']}...
            </div>
"""
        
        html += "        </div>\n"
    
    # Incorrect predictions
    html += "<h2>❌ Incorrect Predictions</h2>"
    if len(incorrect_examples) > 0:
        for ex in incorrect_examples:
            html += f"""
            <div class="example incorrect">
                <div class="header">
                    <div>
                        <strong>Node {ex['node_id']}</strong>
                        <span class="label {ex['true_label'].lower()}">True: {ex['true_label']}</span>
                        <span class="label {ex['predicted_label'].lower()}">Predicted: {ex['predicted_label']}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 12px; color: #666;">Confidence</div>
                        <div style="font-size: 20px; font-weight: bold; color: #ff6b6b;">{ex['confidence']:.1%}</div>
                    </div>
                </div>
                
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {ex['confidence']*100}%; background: #ff6b6b;"></div>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Prob(Fake)</div>
                        <div class="metric-value">{ex['prob_fake']:.3f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Prob(Real)</div>
                        <div class="metric-value">{ex['prob_real']:.3f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Degree</div>
                        <div class="metric-value">{ex['in_degree'] + ex['out_degree']}</div>
                    </div>
                </div>
                
                <div class="network-info">
                    <strong>📊 Network Context:</strong><br>
                    • Connections to this node: {ex['in_degree']} ({ex['in_neighbors_fake_ratio']:.0%} fake)<br>
                    • Connections from this node: {ex['out_degree']} ({ex['out_neighbors_fake_ratio']:.0%} fake)<br>
                    • Attention received: {ex['attention_received']:.2f} | Attention given: {ex['attention_given']:.2f}<br>
                    <span style="color: #d63031;"><strong>⚠️ Possible reason:</strong> Network context conflicts with true label</span>
                </div>
"""
            
            if 'title' in ex:
                html += f"""
                <div class="text-content">
                    <strong>Title:</strong> {ex['title']}<br>
                    <strong>Text:</strong> {ex['text']}...
                </div>
"""
            
            html += "        </div>\n"
    else:
        html += "<p style='text-align: center; color: #6bcf7f; font-weight: bold;'>No incorrect predictions! Perfect classification! 🎉</p>"
    
    html += """
</body>
</html>
"""
    
    with open(os.path.join(output_dir, 'prediction_examples.html'), 'w') as f:
        f.write(html)
    
    print(f"  ✓ Saved prediction examples HTML report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--model-path", type=str, default="experiments/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/node_analysis")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🔬 NODE IMPORTANCE & PREDICTION ANALYSIS")
    print("="*70)
    
    device = get_device() if args.device == "auto" else args.device
    print(f"\n🖥️  Device: {device}")
    
    # Load data and model
    print(f"\n📂 Loading data and model...")
    data = torch.load(args.data_path, weights_only=False).to(device)
    
    model = SimpleGATNode(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=data.y.max().item() + 1,
        num_heads=8,
        dropout=0.3
    ).to(device)
    
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Loaded model from epoch {checkpoint['epoch']}")
    
    # Extract attention weights
    print(f"\n🧠 Extracting attention weights...")
    with torch.no_grad():
        _, attention_weights = model(data.x, data.edge_index, return_attention_weights=True)
    
    # Compute node importance
    print(f"\n📊 Computing node importance metrics...")
    importance_df = compute_node_importance(data, model, attention_weights)
    
    # Save CSV
    importance_df.to_csv(os.path.join(args.output_dir, 'node_importance.csv'), index=False)
    print(f"  ✓ Saved node importance CSV")
    
    # Statistics
    print(f"\n📈 Statistics:")
    print(f"  Total nodes: {len(importance_df)}")
    print(f"  Correct predictions: {importance_df['correct'].sum()} ({importance_df['correct'].mean():.1%})")
    print(f"  Mean confidence: {importance_df['confidence'].mean():.3f}")
    print(f"  Fake news nodes: {(importance_df['true_label'] == 1).sum()}")
    print(f"  Real news nodes: {(importance_df['true_label'] == 0).sum()}")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    plot_node_importance(importance_df, args.output_dir)
    
    # Generate prediction examples
    print(f"\n📝 Generating prediction examples...")
    examples = generate_prediction_examples(importance_df, data, args.output_dir)
    
    # Save examples JSON
    with open(os.path.join(args.output_dir, 'prediction_examples.json'), 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"  ✓ Saved prediction examples JSON")
    
    # Create HTML report
    create_prediction_report(examples, args.output_dir)
    
    # Top nodes analysis
    print(f"\n🏆 Top 10 Most Important Nodes (by attention received):")
    top_nodes = importance_df.nlargest(10, 'attention_received')
    for i, (_, row) in enumerate(top_nodes.iterrows(), 1):
        status = "✓" if row['correct'] else "✗"
        print(f"  {i:2d}. Node {row['node_id']:3d} | {row['label_name']:4s} → {row['prediction_name']:4s} {status} "
              f"| Conf: {row['confidence']:.3f} | Att: {row['attention_received']:.2f} | Deg: {int(row['total_degree'])}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Files generated:")
    print(f"  • node_importance.csv - Full importance metrics")
    print(f"  • node_importance_analysis.png - Visualizations")
    print(f"  • prediction_examples.json - Example predictions")
    print(f"  • prediction_examples.html - Interactive report")
    print(f"\nOpen the HTML report:")
    print(f"  open {args.output_dir}/prediction_examples.html")


if __name__ == "__main__":
    main()
