"""
Utility functions for the project
"""

import os
import json
import yaml
import torch
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        checkpoint_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        device: Device to load model on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    return checkpoint


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(obj: Any, filepath: str):
    """
    Save object using pickle
    
    Args:
        obj: Object to save
        filepath: Path to pickle file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object using pickle
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_device() -> str:
    """
    Get best available device
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directory_structure(base_dir: str = "."):
    """
    Create project directory structure
    
    Args:
        base_dir: Base directory for project
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/graphs",
        "data/cache",
        "experiments",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    print("Directory structure created")


def get_model_summary(model: torch.nn.Module) -> str:
    """
    Get model summary as string
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary string
    """
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total parameters: {count_parameters(model):,}")
    summary.append("\nArchitecture:")
    summary.append(str(model))
    
    return "\n".join(summary)


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Metrics dictionary
        title: Title to display
    """
    print("\n" + "="*50)
    print(title)
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    
    print("="*50 + "\n")


class Logger:
    """Simple file logger"""
    
    def __init__(self, log_file: str):
        """
        Initialize logger
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message: str):
        """
        Log message to file and print
        
        Args:
            message: Message to log
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        Log metrics for an epoch
        
        Args:
            metrics: Metrics dictionary
            epoch: Epoch number
        """
        message = f"Epoch {epoch}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
        )
        self.log(message)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test seed setting
    set_seed(42)
    print("Random seed set to 42")
    
    # Test directory creation
    create_directory_structure(".")
    
    print("\n✅ All utility functions working!")
