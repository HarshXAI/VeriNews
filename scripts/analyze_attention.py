"""
Analyze and visualize GAT attention weights
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device


class SimpleGATNode(torch.nn.Module):
    """Simple GAT for node classification - same as training"""
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


def analyze_attention_patterns(data, attention_weights, layer_idx):
    """Analyze attention patterns for a specific layer"""
    edge_index, alpha = attention_weights[layer_idx]
    
    # Convert to numpy
    edge_index_np = edge_index.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    
    # Average across attention heads
    if len(alpha_np.shape) > 1:
        alpha_np = alpha_np.mean(axis=1)
    
    # Get node labels
    labels = data.y.cpu().numpy()
    
    # Analyze attention by edge type
    fake_to_fake = []
    fake_to_real = []
    real_to_fake = []
    real_to_real = []
    
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        weight = alpha_np[i]
        
        if labels[src] == 1 and labels[dst] == 1:  # fake -> fake
            fake_to_fake.append(weight)
        elif labels[src] == 1 and labels[dst] == 0:  # fake -> real
            fake_to_real.append(weight)
        elif labels[src] == 0 and labels[dst] == 1:  # real -> fake
            real_to_fake.append(weight)
        else:  # real -> real
            real_to_real.append(weight)
    
    stats = {
        'fake_to_fake': {'mean': np.mean(fake_to_fake) if fake_to_fake else 0, 'std': np.std(fake_to_fake) if fake_to_fake else 0, 'count': len(fake_to_fake)},
        'fake_to_real': {'mean': np.mean(fake_to_real) if fake_to_real else 0, 'std': np.std(fake_to_real) if fake_to_real else 0, 'count': len(fake_to_real)},
        'real_to_fake': {'mean': np.mean(real_to_fake) if real_to_fake else 0, 'std': np.std(real_to_fake) if real_to_fake else 0, 'count': len(real_to_fake)},
        'real_to_real': {'mean': np.mean(real_to_real) if real_to_real else 0, 'std': np.std(real_to_real) if real_to_real else 0, 'count': len(real_to_real)},
    }
    
    return stats, (edge_index_np, alpha_np, labels)


def plot_attention_distribution(attention_data, layer_idx, output_dir):
    """Plot attention weight distributions"""
    edge_index_np, alpha_np, labels = attention_data
    
    # Categorize edges
    categories = []
    weights = []
    
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        weight = alpha_np[i]
        
        if labels[src] == 1 and labels[dst] == 1:
            categories.append('Fake → Fake')
        elif labels[src] == 1 and labels[dst] == 0:
            categories.append('Fake → Real')
        elif labels[src] == 0 and labels[dst] == 1:
            categories.append('Real → Fake')
        else:
            categories.append('Real → Real')
        weights.append(weight)
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for violin plot
    data_dict = defaultdict(list)
    for cat, w in zip(categories, weights):
        data_dict[cat].append(w)
    
    # Plot
    positions = []
    data_to_plot = []
    labels_list = []
    
    for idx, (cat, vals) in enumerate(sorted(data_dict.items())):
        if vals:
            positions.append(idx)
            data_to_plot.append(vals)
            labels_list.append(f"{cat}\n(n={len(vals)})")
    
    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
    
    # Customize colors
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4dabf7']
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx % len(colors)])
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list, rotation=0, ha='center')
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(f'Layer {layer_idx + 1} - Attention Weight Distribution by Edge Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attention_layer{layer_idx + 1}_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved attention distribution plot for layer {layer_idx + 1}")


def plot_top_attention_subgraph(data, attention_data, layer_idx, output_dir, top_k=20):
    """Visualize subgraph with top-k attention weights"""
    edge_index_np, alpha_np, labels = attention_data
    
    # Get top-k edges by attention
    top_indices = np.argsort(alpha_np)[-top_k:]
    
    # Create subgraph
    top_edges = edge_index_np[:, top_indices]
    top_weights = alpha_np[top_indices]
    
    # Get unique nodes in subgraph
    unique_nodes = np.unique(top_edges)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with labels
    for node in unique_nodes:
        node_label = 'Fake' if labels[node] == 1 else 'Real'
        G.add_node(int(node), label=node_label, node_id=int(node))
    
    # Add edges with weights
    for i, idx in enumerate(top_indices):
        src, dst = int(top_edges[0, i]), int(top_edges[1, i])
        weight = float(top_weights[i])
        G.add_edge(src, dst, weight=weight)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node colors
    node_colors = ['#ff6b6b' if labels[node] == 1 else '#4dabf7' for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9, ax=ax)
    
    # Draw edges with varying thickness based on attention
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    
    edge_widths = [2 + 8 * (w - min_weight) / (max_weight - min_weight + 1e-8) for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', ax=ax,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, {n: f"{n}" for n in G.nodes()}, 
                           font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(f'Layer {layer_idx + 1} - Top {top_k} Attention Edges\n(Red=Fake, Blue=Real, Edge width=Attention strength)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attention_layer{layer_idx + 1}_top{top_k}_subgraph.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved top-{top_k} attention subgraph for layer {layer_idx + 1}")


def plot_attention_heatmap(stats_all_layers, output_dir):
    """Plot heatmap of attention statistics across layers"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    layer_names = [f'Layer {i+1}' for i in range(len(stats_all_layers))]
    edge_types = ['Fake → Fake', 'Fake → Real', 'Real → Fake', 'Real → Real']
    edge_type_keys = ['fake_to_fake', 'fake_to_real', 'real_to_fake', 'real_to_real']
    
    # Mean attention
    mean_data = np.array([[stats[key]['mean'] for key in edge_type_keys] for stats in stats_all_layers])
    im1 = axes[0].imshow(mean_data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xticks(range(len(layer_names)))
    axes[0].set_xticklabels(layer_names)
    axes[0].set_yticks(range(len(edge_types)))
    axes[0].set_yticklabels(edge_types)
    axes[0].set_title('Mean Attention Weight', fontweight='bold')
    
    # Add values
    for i in range(len(layer_names)):
        for j in range(len(edge_types)):
            text = axes[0].text(i, j, f'{mean_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=axes[0])
    
    # Std attention
    std_data = np.array([[stats[key]['std'] for key in edge_type_keys] for stats in stats_all_layers])
    im2 = axes[1].imshow(std_data.T, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[1].set_xticks(range(len(layer_names)))
    axes[1].set_xticklabels(layer_names)
    axes[1].set_yticks(range(len(edge_types)))
    axes[1].set_yticklabels(edge_types)
    axes[1].set_title('Std Dev of Attention', fontweight='bold')
    
    for i in range(len(layer_names)):
        for j in range(len(edge_types)):
            text = axes[1].text(i, j, f'{std_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=axes[1])
    
    # Count
    count_data = np.array([[stats[key]['count'] for key in edge_type_keys] for stats in stats_all_layers])
    im3 = axes[2].imshow(count_data.T, aspect='auto', cmap='Greens', interpolation='nearest')
    axes[2].set_xticks(range(len(layer_names)))
    axes[2].set_xticklabels(layer_names)
    axes[2].set_yticks(range(len(edge_types)))
    axes[2].set_yticklabels(edge_types)
    axes[2].set_title('Edge Count', fontweight='bold')
    
    for i in range(len(layer_names)):
        for j in range(len(edge_types)):
            text = axes[2].text(i, j, f'{count_data[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle('Attention Patterns Across GAT Layers', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_summary_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved attention summary heatmap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--model-path", type=str, default="experiments/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/attention_analysis")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🔍 ANALYZING GAT ATTENTION WEIGHTS")
    print("="*70)
    
    device = get_device() if args.device == "auto" else args.device
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data and model...")
    data = torch.load(args.data_path, weights_only=False).to(device)
    
    # Load model
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
    print(f"  ✓ Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    
    # Extract attention weights
    print(f"\n🧠 Extracting attention weights...")
    with torch.no_grad():
        _, attention_weights = model(data.x, data.edge_index, return_attention_weights=True)
    
    print(f"  ✓ Extracted attention from {len(attention_weights)} layers")
    
    # Analyze each layer
    print(f"\n📊 Analyzing attention patterns...")
    stats_all_layers = []
    attention_data_all = []
    
    for layer_idx in range(len(attention_weights)):
        print(f"\n  Layer {layer_idx + 1}:")
        stats, attention_data = analyze_attention_patterns(data, attention_weights, layer_idx)
        stats_all_layers.append(stats)
        attention_data_all.append(attention_data)
        
        for edge_type, values in stats.items():
            print(f"    {edge_type:15s}: mean={values['mean']:.4f}, std={values['std']:.4f}, count={values['count']}")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # Distribution plots
    for layer_idx in range(len(attention_weights)):
        plot_attention_distribution(attention_data_all[layer_idx], layer_idx, args.output_dir)
    
    # Top attention subgraphs
    for layer_idx in range(len(attention_weights)):
        plot_top_attention_subgraph(data, attention_data_all[layer_idx], layer_idx, args.output_dir, top_k=20)
    
    # Summary heatmap
    plot_attention_heatmap(stats_all_layers, args.output_dir)
    
    # Save statistics (convert numpy types to Python types)
    stats_dict = {}
    for i, stats in enumerate(stats_all_layers):
        layer_stats = {}
        for edge_type, values in stats.items():
            layer_stats[edge_type] = {
                'mean': float(values['mean']),
                'std': float(values['std']),
                'count': int(values['count'])
            }
        stats_dict[f'layer_{i+1}'] = layer_stats
    
    with open(os.path.join(args.output_dir, 'attention_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    
    # Print key insights
    print("\n🔑 Key Insights:")
    
    for layer_idx, stats in enumerate(stats_all_layers):
        print(f"\n  Layer {layer_idx + 1}:")
        
        # Find edge type with highest attention
        max_type = max(stats.items(), key=lambda x: x[1]['mean'])
        print(f"    • Highest attention: {max_type[0].replace('_', ' → ').title()} (mean={max_type[1]['mean']:.4f})")
        
        # Find most common edge type
        max_count_type = max(stats.items(), key=lambda x: x[1]['count'])
        print(f"    • Most common edges: {max_count_type[0].replace('_', ' → ').title()} ({max_count_type[1]['count']} edges)")


if __name__ == "__main__":
    main()
