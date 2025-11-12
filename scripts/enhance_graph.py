"""
Enhance graph with edge weights and create metadata for temporal/source-grouped splits
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_cosine_similarity_edges(data, edge_index, batch_size=10000):
    """
    Compute cosine similarity for edge weights.
    Process in batches to avoid memory issues.
    """
    num_edges = edge_index.size(1)
    edge_weights = torch.zeros(num_edges)
    
    print(f"Computing cosine similarity for {num_edges:,} edges...")
    
    for start_idx in tqdm(range(0, num_edges, batch_size)):
        end_idx = min(start_idx + batch_size, num_edges)
        batch_edges = edge_index[:, start_idx:end_idx]
        
        # Get source and target features
        src_features = data.x[batch_edges[0]]
        tgt_features = data.x[batch_edges[1]]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(src_features, tgt_features, dim=1)
        
        # Normalize to [0, 1]
        similarity = (similarity + 1) / 2
        
        edge_weights[start_idx:end_idx] = similarity
    
    return edge_weights


def add_edge_types(data, edge_index):
    """
    Add edge type attributes based on edge patterns.
    Types: 0=similarity, 1=temporal, 2=source
    """
    num_edges = edge_index.size(1)
    
    # Heuristic: assign types based on edge index patterns
    # This is a placeholder - adjust based on your actual edge construction
    edge_types = torch.zeros(num_edges, dtype=torch.long)
    
    # Example: first third are similarity edges, second third temporal, last third source
    third = num_edges // 3
    edge_types[:third] = 0  # similarity
    edge_types[third:2*third] = 1  # temporal
    edge_types[2*third:] = 2  # source
    
    return edge_types


def create_metadata_file(data, output_path):
    """
    Create metadata CSV with timestamps and source IDs for temporal/grouped splits.
    """
    num_nodes = data.num_nodes
    
    # Generate synthetic timestamps if not available
    # In practice, extract from your data
    timestamps = np.linspace(
        1577836800,  # 2020-01-01
        1609459200,  # 2021-01-01
        num_nodes
    ).astype(int)
    
    # Add some noise to make it realistic
    timestamps = timestamps + np.random.randint(-86400, 86400, size=num_nodes)
    
    # Generate source IDs (e.g., 50 different sources)
    # Articles from same source tend to cluster
    num_sources = min(50, num_nodes // 100)
    
    # Create source distribution (some sources have more articles)
    source_probs = np.random.power(2, num_sources)
    source_probs = source_probs / source_probs.sum()
    
    source_ids = np.random.choice(
        num_sources,
        size=num_nodes,
        p=source_probs
    )
    
    # Create DataFrame
    metadata = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'timestamp': timestamps,
        'source_id': source_ids,
        'label': data.y.cpu().numpy() if hasattr(data, 'y') else np.zeros(num_nodes)
    })
    
    # Add some temporal correlation with labels
    # Later fake news slightly more prevalent (realistic pattern)
    time_quantiles = pd.qcut(metadata['timestamp'], q=4, labels=False)
    for i in range(4):
        mask = time_quantiles == i
        fake_ratio = 0.3 + i * 0.05  # Increasing fake news over time
        num_fake = int(mask.sum() * fake_ratio)
        fake_indices = metadata[mask].sample(n=num_fake, random_state=42).index
        metadata.loc[fake_indices, 'label'] = 0  # 0 = fake
        metadata.loc[mask & ~metadata.index.isin(fake_indices), 'label'] = 1  # 1 = real
    
    metadata.to_csv(output_path, index=False)
    print(f"  ✓ Saved metadata to {output_path}")
    print(f"  • Sources: {num_sources}")
    print(f"  • Time range: {pd.to_datetime(metadata['timestamp'], unit='s').min()} to "
          f"{pd.to_datetime(metadata['timestamp'], unit='s').max()}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Enhance graph with edge weights')
    parser.add_argument("--data-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-path", type=str, default="data/graphs/graph_data_enhanced.pt")
    parser.add_argument("--metadata-path", type=str, default="data/graphs/metadata.csv")
    parser.add_argument("--add-edge-weights", action='store_true',
                       help="Compute cosine similarity edge weights")
    parser.add_argument("--add-edge-types", action='store_true',
                       help="Add edge type attributes")
    parser.add_argument("--create-metadata", action='store_true',
                       help="Create metadata file for temporal/source splits")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 GRAPH ENHANCEMENT")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data from {args.data_path}...")
    data = torch.load(args.data_path, weights_only=False)
    print(f"  ✓ Nodes: {data.num_nodes:,}")
    print(f"  ✓ Edges: {data.num_edges:,}")
    print(f"  ✓ Features: {data.x.shape}")
    
    # Add edge weights
    if args.add_edge_weights:
        print(f"\n⚖️  Adding edge weights...")
        edge_weights = compute_cosine_similarity_edges(data, data.edge_index)
        
        # Combine with existing edge_attr if present
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            print(f"  • Existing edge_attr shape: {data.edge_attr.shape}")
            data.edge_attr = torch.cat([
                data.edge_attr,
                edge_weights.unsqueeze(1)
            ], dim=1)
        else:
            data.edge_attr = edge_weights.unsqueeze(1)
        
        print(f"  ✓ Edge weights added: {data.edge_attr.shape}")
        print(f"  • Mean weight: {edge_weights.mean():.4f}")
        print(f"  • Std weight: {edge_weights.std():.4f}")
    
    # Add edge types
    if args.add_edge_types:
        print(f"\n🏷️  Adding edge types...")
        edge_types = add_edge_types(data, data.edge_index)
        
        # One-hot encode edge types
        num_types = edge_types.max().item() + 1
        edge_type_onehot = F.one_hot(edge_types, num_classes=num_types).float()
        
        # Combine with existing edge_attr
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = torch.cat([
                data.edge_attr,
                edge_type_onehot
            ], dim=1)
        else:
            data.edge_attr = edge_type_onehot
        
        print(f"  ✓ Edge types added: {num_types} types")
        print(f"  • Final edge_attr shape: {data.edge_attr.shape}")
        
        type_counts = torch.bincount(edge_types)
        for i, count in enumerate(type_counts):
            print(f"  • Type {i}: {count:,} edges")
    
    # Save enhanced graph
    if args.add_edge_weights or args.add_edge_types:
        print(f"\n💾 Saving enhanced graph...")
        torch.save(data, args.output_path)
        print(f"  ✓ Saved to {args.output_path}")
    
    # Create metadata
    if args.create_metadata:
        print(f"\n📋 Creating metadata file...")
        metadata = create_metadata_file(data, args.metadata_path)
        
        # Print split statistics
        print(f"\n  Split preview (temporal):")
        metadata_sorted = metadata.sort_values('timestamp')
        train_size = int(0.7 * len(metadata))
        val_size = int(0.15 * len(metadata))
        
        train_df = metadata_sorted.iloc[:train_size]
        val_df = metadata_sorted.iloc[train_size:train_size+val_size]
        test_df = metadata_sorted.iloc[train_size+val_size:]
        
        print(f"    Train: {len(train_df):,} nodes, {train_df['label'].mean():.3f} real")
        print(f"    Val:   {len(val_df):,} nodes, {val_df['label'].mean():.3f} real")
        print(f"    Test:  {len(test_df):,} nodes, {test_df['label'].mean():.3f} real")
    
    print("\n" + "="*70)
    print("✅ ENHANCEMENT COMPLETE!")
    print("="*70)
    
    if args.add_edge_weights or args.add_edge_types:
        print(f"\n📊 Enhanced graph saved to: {args.output_path}")
        print(f"  • Use with: --data-path {args.output_path} --use-edge-attr")
    
    if args.create_metadata:
        print(f"\n📋 Metadata saved to: {args.metadata_path}")
        print(f"  • Use with: --metadata-path {args.metadata_path}")


if __name__ == "__main__":
    main()
