"""
Fake News Propagation Detection using Graph Attention Networks
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.models.gat_model import FakeNewsGAT
from src.training.trainer import GATTrainer

__all__ = [
    "FakeNewsGAT",
    "GATTrainer",
]
