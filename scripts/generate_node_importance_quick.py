"""
Quick script to generate node importance data for dashboard
Matches the actual saved model architecture
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Model definition matching the saved checkpoint
class SimpleGATNode(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, dropout=0.6):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        
    def forward(self, x, edge_index, return_attention_weights=False):
        attention_weights = []
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if return_attention_weights:
                x, (edge_idx, alpha) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, alpha))
            else:
                x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # ELU for all but last layer
                x = F.elu(x)
        
        if return_attention_weights:
            return F.log_softmax(x, dim=1), attention_weights
        else:
            return F.log_softmax(x, dim=1)


def main():
    print("=" * 70)
    print("🔬 GENERATING NODE IMPORTANCE DATA")
    print("=" * 70)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n📂 Loading graph data...")
    data = torch.load('data/graphs_full/graph_data_enriched.pt', weights_only=False)
    data = data.to(device)
    
    print(f"   Nodes: {data.num_nodes:,}")
    print(f"   Edges: {data.edge_index.shape[1]:,}")
    print(f"   Features: {data.x.shape[1]}")
    
    # Load model
    print("\n🤖 Loading trained model...")
    model = SimpleGATNode(
        in_channels=data.x.shape[1],
        hidden_channels=256,  # Match training configuration
        out_channels=2,
        num_heads=8,
        dropout=0.6
    ).to(device)
    
    checkpoint = torch.load('experiments/models_fullscale/gat_model_best_scaled.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("   ✓ Model loaded successfully")
    
    # Get predictions and attention weights
    print("\n🔍 Computing predictions and attention weights...")
    with torch.no_grad():
        out, attention_weights = model(data.x, data.edge_index, return_attention_weights=True)
        probs = torch.exp(out)
        pred = out.argmax(dim=1)
        confidence = probs.max(dim=1)[0]
    
    print(f"   ✓ Processed {data.num_nodes:,} nodes")
    
    # Compute degree centrality
    print("\n📊 Computing node metrics...")
    from torch_geometric.utils import degree
    
    edge_index = data.edge_index.cpu()
    in_degree = degree(edge_index[1], num_nodes=data.num_nodes).cpu().numpy()
    out_degree = degree(edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    total_degree = in_degree + out_degree
    
    # Compute attention scores
    attention_received = np.zeros(data.num_nodes)
    attention_given = np.zeros(data.num_nodes)
    attention_count_received = np.zeros(data.num_nodes)
    attention_count_given = np.zeros(data.num_nodes)
    
    for layer_idx, (edge_idx, alpha) in enumerate(attention_weights):
        edge_idx_np = edge_idx.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        
        # Average across heads if multi-head
        if len(alpha_np.shape) > 1:
            alpha_np = alpha_np.mean(axis=1)
        
        # Sum attention for each node
        for i in range(edge_idx_np.shape[1]):
            src, dst = edge_idx_np[0, i], edge_idx_np[1, i]
            attention_received[dst] += alpha_np[i]
            attention_given[src] += alpha_np[i]
            attention_count_received[dst] += 1
            attention_count_given[src] += 1
    
    # Normalize by number of edges (average attention per edge)
    attention_received = np.where(attention_count_received > 0, 
                                  attention_received / attention_count_received, 
                                  0)
    attention_given = np.where(attention_count_given > 0, 
                              attention_given / attention_count_given, 
                              0)
    
    print("   ✓ Computed degree, attention scores")
    
    # Create output directory
    output_dir = Path('experiments/node_importance')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full metrics
    print("\n💾 Saving node importance data...")
    
    importance_df = pd.DataFrame({
        'node_id': range(data.num_nodes),
        'degree': total_degree,
        'attention_score': attention_received,
        'attention_given': attention_given,
        'confidence': confidence.cpu().numpy(),
        'predicted_label': pred.cpu().numpy(),
        'true_label': data.y.cpu().numpy()
    })
    
    importance_df.to_csv(output_dir / 'node_importance_metrics.csv', index=False)
    print(f"   ✓ Saved: {output_dir / 'node_importance_metrics.csv'}")
    
    # Save top influential nodes (for presentation)
    top_nodes = importance_df.nlargest(100, 'degree')
    top_nodes.to_csv(output_dir / 'top_influential_nodes.csv', index=False)
    print(f"   ✓ Saved: {output_dir / 'top_influential_nodes.csv'}")
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   Degree range: {total_degree.min():.0f} - {total_degree.max():.0f}")
    print(f"   Avg degree: {total_degree.mean():.2f}")
    print(f"   Attention received range: {attention_received.min():.4f} - {attention_received.max():.4f}")
    print(f"   Avg attention: {attention_received.mean():.4f}")
    print(f"   Avg confidence: {confidence.mean().item():.4f}")
    
    # Model performance
    correct = (pred == data.y).sum().item()
    accuracy = correct / data.num_nodes
    print(f"\n   Model accuracy: {accuracy:.4f} ({correct:,}/{data.num_nodes:,})")
    
    print("\n" + "=" * 70)
    print("✅ NODE IMPORTANCE DATA GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  • {output_dir / 'node_importance_metrics.csv'}")
    print(f"  • {output_dir / 'top_influential_nodes.csv'}")
    print("\nReady for dashboard! 🎉")


if __name__ == '__main__':
    main()
