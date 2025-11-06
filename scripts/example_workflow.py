"""
Complete end-to-end example workflow for fake news detection

This script demonstrates the full pipeline from data loading to model evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data import FakeNewsNetLoader, preprocess_news_data, preprocess_social_data
from src.features import TextEmbedder, UserFeatureEncoder, PropagationGraphBuilder
from src.models import FakeNewsGAT
from src.training import GATTrainer
from src.evaluation import ModelEvaluator, AttentionAnalyzer
from src.visualization import GraphVisualizer, MetricsVisualizer


def check_data_exists(data_dir: str = "data/processed") -> bool:
    """Check if processed data exists"""
    news_path = os.path.join(data_dir, "news_processed.parquet")
    social_path = os.path.join(data_dir, "social_processed.parquet")
    return os.path.exists(news_path) and os.path.exists(social_path)


def load_processed_data(data_dir: str = "data/processed"):
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    news_df = pd.read_parquet(os.path.join(data_dir, "news_processed.parquet"))
    social_df = pd.read_parquet(os.path.join(data_dir, "social_processed.parquet"))
    
    print(f"Loaded {len(news_df)} news articles")
    print(f"Loaded {len(social_df)} social posts")
    
    return news_df, social_df


def create_graph_data(news_df, social_df, embedder, max_samples=1000):
    """
    Create PyTorch Geometric Data objects from preprocessed data
    
    Args:
        news_df: Preprocessed news DataFrame
        social_df: Preprocessed social DataFrame
        embedder: Text embedder
        max_samples: Maximum number of samples to process
        
    Returns:
        List of Data objects
    """
    print("\nCreating graph data...")
    
    # Limit samples for demonstration
    news_df = news_df.head(max_samples)
    
    graph_data_list = []
    
    for idx, news_row in news_df.iterrows():
        news_id = news_row['news_id']
        label = news_row['label_encoded']
        
        # Get social posts for this news
        news_social = social_df[social_df['news_id'] == news_id]
        
        if len(news_social) < 3:  # Skip if too few interactions
            continue
        
        # Generate embeddings
        text = news_row.get('text_clean', news_row.get('title_clean', ''))
        if not text or len(text) < 10:
            continue
        
        text_embedding = embedder.embed_single(text)
        
        # Create simple features (for demonstration)
        num_nodes = min(len(news_social), 50)  # Limit for memory
        node_features = []
        
        # Add news node
        node_features.append(text_embedding)
        
        # Add user nodes (simplified)
        for i in range(num_nodes - 1):
            # Use text embedding as placeholder
            node_features.append(text_embedding)
        
        # Stack features
        x = torch.tensor(np.vstack(node_features), dtype=torch.float)
        
        # Create simple edge index (star graph from news to users)
        edge_index = []
        for i in range(1, num_nodes):
            edge_index.append([0, i])  # News -> User
            edge_index.append([i, 0])  # User -> News
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create label
        y = torch.tensor([label], dtype=torch.long)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        graph_data_list.append(data)
        
        if len(graph_data_list) % 100 == 0:
            print(f"Processed {len(graph_data_list)} graphs...")
    
    print(f"Created {len(graph_data_list)} graph data objects")
    return graph_data_list


def split_data(graph_data_list, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    n = len(graph_data_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Shuffle
    indices = np.random.permutation(n)
    
    train_data = [graph_data_list[i] for i in indices[:n_train]]
    val_data = [graph_data_list[i] for i in indices[n_train:n_train+n_val]]
    test_data = [graph_data_list[i] for i in indices[n_train+n_val:]]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def main():
    """Main workflow"""
    print("="*60)
    print("Fake News Detection - Complete Workflow Example")
    print("="*60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if data exists
    if not check_data_exists():
        print("\n❌ Processed data not found!")
        print("Please run: python scripts/preprocess_data.py")
        return
    
    # Load data
    news_df, social_df = load_processed_data()
    
    # Initialize embedder
    print("\nInitializing text embedder...")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    embedder = TextEmbedder(device=device)
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    # Create graph data (limited samples for demo)
    graph_data_list = create_graph_data(
        news_df, 
        social_df, 
        embedder, 
        max_samples=500  # Limit for demonstration
    )
    
    if len(graph_data_list) == 0:
        print("\n❌ No graph data created!")
        return
    
    # Split data
    train_data, val_data, test_data = split_data(graph_data_list)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating GAT model...")
    model = FakeNewsGAT(
        in_channels=embedder.embedding_dim,
        hidden_channels=128,
        out_channels=2,
        num_layers=3,
        num_heads=8,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = GATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        device=device,
        checkpoint_dir="experiments"
    )
    
    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    trainer.train(epochs=20, early_stopping_patience=5)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate_and_report(test_loader)
    
    # Save results
    import json
    results = {
        'test_metrics': metrics,
        'model_config': {
            'in_channels': embedder.embedding_dim,
            'hidden_channels': 128,
            'num_layers': 3,
            'num_heads': 8
        },
        'data_split': {
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        }
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to outputs/results.json")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = MetricsVisualizer()
    visualizer.plot_metrics_comparison(
        metrics,
        save_path='outputs/metrics_comparison.png'
    )
    
    print("\n" + "="*60)
    print("Workflow complete!")
    print("="*60)
    print(f"\nCheckpoints saved in: experiments/")
    print(f"Results saved in: outputs/")


if __name__ == "__main__":
    main()
