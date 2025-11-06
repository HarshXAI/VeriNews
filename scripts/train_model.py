"""
Train GAT model for fake news detection

Usage:
    python scripts/train_model.py --config configs/model_config.yaml
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FakeNewsGAT
from src.training import GATTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train GAT model for fake news detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/graphs/processed_graphs.pt",
        help="Path to preprocessed graph data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments",
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load data
    print(f"Loading data from {args.data}...")
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        print("\nPlease run data preprocessing first:")
        print("  python scripts/preprocess_data.py")
        sys.exit(1)
    
    # TODO: Implement data loading
    # data = torch.load(args.data)
    # train_loader, val_loader, test_loader = create_dataloaders(data)
    
    # Create model
    print("Creating model...")
    model_config = config['model']['architecture']
    model = FakeNewsGAT(
        in_channels=model_config['in_channels'],
        hidden_channels=model_config['hidden_channels'],
        out_channels=model_config['out_channels'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        concat_heads=model_config['concat_heads'],
        use_batch_norm=config['model']['output']['use_batch_norm']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # TODO: Create trainer and train
    # training_config = config['training']
    # trainer = GATTrainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     learning_rate=training_config['learning_rate'],
    #     weight_decay=training_config['weight_decay'],
    #     checkpoint_dir=args.output
    # )
    
    # trainer.train(
    #     epochs=training_config['epochs'],
    #     early_stopping_patience=training_config['regularization']['early_stopping']['patience']
    # )
    
    print("\nTraining script template created!")
    print("Complete the data loading implementation to start training.")


if __name__ == "__main__":
    main()
