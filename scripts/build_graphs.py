"""
Build and save graph data for training

Usage:
    python scripts/build_graphs.py --input data/processed --output data/graphs
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from tqdm import tqdm

from src.features import TextEmbedder, PropagationGraphBuilder
from src.utils import save_pickle, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Build propagation graphs from preprocessed data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed",
        help="Input directory with preprocessed data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/graphs",
        help="Output directory for graph data"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of news articles to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for embeddings (cuda/mps/cpu/auto)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    news_path = os.path.join(args.input, "news_processed.parquet")
    social_path = os.path.join(args.input, "social_processed.parquet")
    
    if not os.path.exists(news_path) or not os.path.exists(social_path):
        print(f"Error: Preprocessed data not found in {args.input}")
        print("\nPlease run preprocessing first:")
        print("  python scripts/preprocess_data.py")
        sys.exit(1)
    
    news_df = pd.read_parquet(news_path)
    social_df = pd.read_parquet(social_path)
    
    print(f"Loaded {len(news_df)} news articles")
    print(f"Loaded {len(social_df)} social posts")
    
    # Limit samples
    if len(news_df) > args.max_samples:
        print(f"\nLimiting to {args.max_samples} samples for demonstration")
        news_df = news_df.sample(n=args.max_samples, random_state=42)
    
    # Initialize embedder
    print("\nInitializing text embedder...")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    embedder = TextEmbedder(device=device)
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    # Generate text embeddings
    print("\nGenerating text embeddings...")
    # Use title_clean since we don't have full text content in CSV format
    texts = news_df['title_clean'].fillna(news_df['title']).tolist()
    embeddings = embedder.embed_texts(texts, batch_size=32, show_progress=True)
    
    # Save embeddings
    embeddings_path = os.path.join(args.output, "text_embeddings.pt")
    torch.save(torch.tensor(embeddings), embeddings_path)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Build propagation graphs
    print("\nBuilding propagation graphs...")
    builder = PropagationGraphBuilder()
    
    # Filter social data to only include tweets/retweets for our news sample
    news_ids = set(news_df['id'].tolist())
    social_filtered = social_df[social_df['news_id'].isin(news_ids)]
    
    print(f"Filtered to {len(social_filtered)} relevant social posts")
    
    # Build graph
    if len(social_filtered) > 0:
        graph = builder.build_graph(social_filtered, social_filtered)
        
        # Get statistics
        stats = builder.get_statistics()
        print("\nGraph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save graph
        graph_path = os.path.join(args.output, "propagation_graph.pkl")
        save_pickle(graph, graph_path)
        print(f"\nSaved propagation graph to {graph_path}")
    else:
        print("\n⚠️  Warning: No social posts found for graph construction")
    
    # Save metadata
    metadata = {
        'num_news': len(news_df),
        'num_social': len(social_filtered),
        'embedding_dim': embedder.embedding_dim,
        'device': device,
        'max_samples': args.max_samples
    }
    
    import json
    metadata_path = os.path.join(args.output, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Graph construction complete!")
    print(f"Output saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Run training: python scripts/train_model.py")
    print("  2. Or run example workflow: python scripts/example_workflow.py")


if __name__ == "__main__":
    main()
