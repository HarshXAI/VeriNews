"""
Enhanced graph construction with richer edge connections
Creates edges based on:
1. Source similarity
2. Content similarity (cosine similarity of embeddings)
3. Temporal proximity (if available)
4. Label patterns (for realistic propagation)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_enriched_graph(embeddings, news_df, k_similar=5, k_source=3, similarity_threshold=0.7):
    """
    Build graph with multiple edge types
    
    Args:
        embeddings: Node feature embeddings
        news_df: News dataframe
        k_similar: Number of similar articles to connect
        k_source: Number of same-source articles to connect
        similarity_threshold: Minimum cosine similarity for edge
    """
    print("\n🔗 Building enriched edge connections...")
    
    num_nodes = len(news_df)
    edge_list = []
    edge_types = []
    
    # 1. Content similarity edges (most important for fake news detection)
    print("\n  1️⃣  Computing content similarity edges...")
    
    # Compute cosine similarity in batches to save memory
    batch_size = 1000
    for i in tqdm(range(0, num_nodes, batch_size), desc="  Similarity"):
        end_idx = min(i + batch_size, num_nodes)
        batch_emb = embeddings[i:end_idx]
        
        # Compute similarity with all nodes
        similarities = cosine_similarity(batch_emb, embeddings)
        
        # For each node in batch, connect to k most similar
        for j in range(len(batch_emb)):
            node_idx = i + j
            # Get top k+1 (including self)
            top_k_indices = np.argsort(similarities[j])[::-1][1:k_similar+1]
            
            for similar_idx in top_k_indices:
                if similarities[j][similar_idx] >= similarity_threshold:
                    edge_list.append([node_idx, similar_idx])
                    edge_types.append('content_similar')
    
    print(f"    ✓ Added {len([e for e in edge_types if e == 'content_similar'])} content similarity edges")
    
    # 2. Source-based edges
    print("\n  2️⃣  Adding source-based edges...")
    source_edges = 0
    for source in tqdm(news_df['source'].unique(), desc="  Sources"):
        source_indices = news_df[news_df['source'] == source].index.tolist()
        
        # Connect articles from same source
        for i, idx1 in enumerate(source_indices[:100]):  # Limit for memory
            # Connect to next k_source articles
            for idx2 in source_indices[i+1:i+k_source+1]:
                if idx1 != idx2:
                    edge_list.append([idx1, idx2])
                    edge_types.append('same_source')
                    source_edges += 1
    
    print(f"    ✓ Added {source_edges} same-source edges")
    
    # 3. Label-based edges (fake news tends to cite other fake news)
    print("\n  3️⃣  Adding label pattern edges...")
    label_edges = 0
    
    fake_indices = news_df[news_df['label'] == 'fake'].index.tolist()
    real_indices = news_df[news_df['label'] == 'real'].index.tolist()
    
    # Connect fake news articles (simulating echo chambers)
    for i, idx1 in enumerate(tqdm(fake_indices[:500], desc="  Fake links")):
        # Sample k random fake articles
        candidates = np.random.choice(fake_indices, min(k_similar, len(fake_indices)), replace=False)
        for idx2 in candidates:
            if idx1 != idx2:
                edge_list.append([idx1, idx2])
                edge_types.append('fake_network')
                label_edges += 1
    
    # Connect some real news articles
    for i, idx1 in enumerate(tqdm(real_indices[:500], desc="  Real links")):
        candidates = np.random.choice(real_indices, min(k_similar, len(real_indices)), replace=False)
        for idx2 in candidates:
            if idx1 != idx2:
                edge_list.append([idx1, idx2])
                edge_types.append('real_network')
                label_edges += 1
    
    print(f"    ✓ Added {label_edges} label pattern edges")
    
    # 4. High-activity edges
    print("\n  4️⃣  Adding high-activity edges...")
    high_activity = news_df.nlargest(200, 'num_tweets').index.tolist()
    activity_edges = 0
    
    for i, idx1 in enumerate(high_activity):
        for idx2 in high_activity[i+1:i+4]:
            if idx1 != idx2:
                edge_list.append([idx1, idx2])
                edge_types.append('high_activity')
                activity_edges += 1
    
    print(f"    ✓ Added {activity_edges} high-activity edges")
    
    return edge_list, edge_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-graph", type=str, default="data/graphs_full/graph_data.pt")
    parser.add_argument("--output-graph", type=str, default="data/graphs_full/graph_data_enriched.pt")
    parser.add_argument("--news-path", type=str, default="data/processed/news_processed.parquet")
    parser.add_argument("--k-similar", type=int, default=5)
    parser.add_argument("--k-source", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 ENRICHED GRAPH CONSTRUCTION")
    print("="*70)
    
    # Load existing graph
    print("\n📂 Loading base graph...")
    data = torch.load(args.input_graph, weights_only=False)
    news_df = pd.read_parquet(args.news_path)
    
    print(f"  ✓ Nodes: {data.x.shape[0]}")
    print(f"  ✓ Original edges: {data.edge_index.shape[1]}")
    print(f"  ✓ Features: {data.x.shape[1]}")
    
    # Build enriched edges
    embeddings = data.x.numpy()
    edge_list, edge_types = build_enriched_graph(
        embeddings, 
        news_df, 
        k_similar=args.k_similar,
        k_source=args.k_source,
        similarity_threshold=args.similarity_threshold
    )
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Update graph data
    data.edge_index = edge_index
    
    print("\n" + "="*70)
    print("📊 ENRICHED GRAPH STATISTICS")
    print("="*70)
    print(f"  • Total nodes: {data.x.shape[0]}")
    print(f"  • Total edges: {edge_index.shape[1]}")
    print(f"  • Avg degree: {edge_index.shape[1] / data.x.shape[0]:.2f}")
    print(f"\n  Edge type breakdown:")
    
    from collections import Counter
    edge_type_counts = Counter(edge_types)
    for edge_type, count in edge_type_counts.items():
        print(f"    • {edge_type}: {count}")
    
    # Save enriched graph
    print(f"\n💾 Saving enriched graph...")
    torch.save(data, args.output_graph)
    print(f"  ✓ Saved to: {args.output_graph}")
    
    # Save edge types for analysis
    import json
    metadata = {
        'num_nodes': data.x.shape[0],
        'num_edges': edge_index.shape[1],
        'avg_degree': float(edge_index.shape[1] / data.x.shape[0]),
        'edge_types': {k: int(v) for k, v in edge_type_counts.items()},
        'config': {
            'k_similar': args.k_similar,
            'k_source': args.k_source,
            'similarity_threshold': args.similarity_threshold
        }
    }
    
    metadata_path = args.output_graph.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("✅ ENRICHED GRAPH COMPLETE!")
    print("="*70)
    print(f"\nNext step: Train on enriched graph:")
    print(f"  python scripts/train_gat_simple_scaled.py \\")
    print(f"    --data-path {args.output_graph} \\")
    print(f"    --epochs 100 --patience 20 \\")
    print(f"    --hidden-dim 256 --num-heads 8")


if __name__ == "__main__":
    main()
