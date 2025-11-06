"""
Build simplified propagation graphs for demonstration

Since we only have tweet IDs (not full tweet data), we'll create
a simplified graph structure based on news articles and tweet counts.
"""

import argparse
import os
import sys
from pathlib import Path
import pickle

import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.embeddings import TextEmbedder
from src.utils import get_device


def create_news_graph(news_df, social_df):
    """Create a simple graph where news articles are nodes"""
    G = nx.DiGraph()
    
    # Add news nodes
    for idx, row in news_df.iterrows():
        G.add_node(
            row['id'],
            label=row['label'],
            label_encoded=row['label_encoded'],
            source=row['source'],
            num_tweets=row['num_tweets'],
            title=row['title'][:100]  # Truncate for memory
        )
    
    # Create edges between news articles that share similar sources
    # Group by source
    for source in news_df['source'].unique():
        source_news = news_df[news_df['source'] == source]['id'].tolist()
        # Connect news from same source (simplified propagation)
        for i, news1 in enumerate(source_news[:50]):  # Limit for demo
            for news2 in source_news[i+1:i+6]:  # Connect to next 5
                if news1 != news2:
                    G.add_edge(news1, news2, edge_type='same_source')
    
    # Add edges based on tweet activity
    tweet_counts = news_df.set_index('id')['num_tweets'].to_dict()
    high_activity = [nid for nid, count in tweet_counts.items() if count > 50]
    
    # Connect high-activity news (simulating viral spread)
    for i, news1 in enumerate(high_activity[:20]):
        for news2 in high_activity[i+1:i+4]:
            if news1 != news2 and not G.has_edge(news1, news2):
                G.add_edge(news1, news2, edge_type='high_activity')
    
    return G


def main():
    parser = argparse.ArgumentParser(description="Build simplified propagation graphs")
    parser.add_argument("--input", type=str, default="data/processed", help="Input directory")
    parser.add_argument("--output", type=str, default="data/graphs", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to process")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*70)
    print("BUILDING SIMPLIFIED PROPAGATION GRAPHS")
    print("="*70)
    
    # Load data
    print("\nLoading preprocessed data...")
    news_df = pd.read_parquet(os.path.join(args.input, "news_processed.parquet"))
    social_df = pd.read_parquet(os.path.join(args.input, "social_processed.parquet"))
    
    print(f"  ✓ Loaded {len(news_df)} news articles")
    print(f"  ✓ Loaded {len(social_df)} social mappings")
    
    # Sample if needed
    if len(news_df) > args.max_samples:
        print(f"\nLimiting to {args.max_samples} samples...")
        news_df = news_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        news_ids = set(news_df['id'].tolist())
        social_df = social_df[social_df['news_id'].isin(news_ids)]
    
    # Generate embeddings
    print("\n" + "="*70)
    print("GENERATING TEXT EMBEDDINGS")
    print("="*70)
    
    device = get_device() if args.device == "auto" else args.device
    print(f"  Device: {device}")
    
    embedder = TextEmbedder(device=device)
    print(f"  Embedding dimension: {embedder.embedding_dim}")
    
    texts = news_df['title_clean'].fillna(news_df['title']).tolist()
    print(f"  Processing {len(texts)} texts...")
    
    embeddings = embedder.embed_texts(texts, batch_size=16, show_progress=True)
    
    # Save embeddings
    embeddings_path = os.path.join(args.output, "text_embeddings.pt")
    torch.save(torch.tensor(embeddings), embeddings_path)
    print(f"  ✓ Saved embeddings: {embeddings_path}")
    print(f"    Shape: {embeddings.shape}")
    
    # Build graph
    print("\n" + "="*70)
    print("BUILDING PROPAGATION GRAPH")
    print("="*70)
    
    G = create_news_graph(news_df, social_df)
    
    print(f"  ✓ Created graph:")
    print(f"    - Nodes: {G.number_of_nodes()}")
    print(f"    - Edges: {G.number_of_edges()}")
    print(f"    - Density: {nx.density(G):.4f}")
    
    if G.number_of_edges() > 0:
        print(f"    - Weakly connected: {nx.is_weakly_connected(G)}")
        print(f"    - Components: {nx.number_weakly_connected_components(G)}")
    
    # Save graph
    graph_path = os.path.join(args.output, "propagation_graph.pkl")
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"  ✓ Saved graph: {graph_path}")
    
    # Save metadata
    metadata = {
        'num_samples': len(news_df),
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'embedding_dim': embedder.embedding_dim,
        'num_fake': int((news_df['label'] == 'fake').sum()),
        'num_real': int((news_df['label'] == 'real').sum()),
        'sources': news_df['source'].value_counts().to_dict()
    }
    
    metadata_path = os.path.join(args.output, "metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {metadata_path}")
    
    # Create node features matrix
    print("\n" + "="*70)
    print("CREATING NODE FEATURES")
    print("="*70)
    
    # Map news IDs to indices
    news_id_to_idx = {nid: idx for idx, nid in enumerate(news_df['id'])}
    
    # Create feature matrix: [num_nodes, embedding_dim]
    node_features = embeddings
    
    # Create labels
    labels = torch.tensor(news_df['label_encoded'].values, dtype=torch.long)
    
    # Save PyTorch Geometric compatible data
    from torch_geometric.data import Data
    
    # Convert graph edges to tensor
    edge_list = list(G.edges())
    if edge_list:
        edge_index = torch.tensor([
            [news_id_to_idx[e[0]] for e in edge_list],
            [news_id_to_idx[e[1]] for e in edge_list]
        ], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=labels
    )
    
    pyg_path = os.path.join(args.output, "graph_data.pt")
    torch.save(data, pyg_path)
    print(f"  ✓ Saved PyTorch Geometric data: {pyg_path}")
    print(f"    - Features shape: {data.x.shape}")
    print(f"    - Edges shape: {data.edge_index.shape}")
    print(f"    - Labels shape: {data.y.shape}")
    
    print("\n" + "="*70)
    print("✅ GRAPH CONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"\nOutput saved to: {args.output}")
    print("\nNext step: Train the GAT model")
    print("  python scripts/train_model.py")


if __name__ == "__main__":
    main()
