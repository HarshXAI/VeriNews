"""
Unit tests for GAT model
"""

import unittest
import torch
from src.models import FakeNewsGAT


class TestFakeNewsGAT(unittest.TestCase):
    """Test GAT model"""
    
    def setUp(self):
        self.model = FakeNewsGAT(
            in_channels=768,
            hidden_channels=64,
            out_channels=2,
            num_layers=2,
            num_heads=4
        )
    
    def test_model_creation(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.in_channels, 768)
        self.assertEqual(self.model.hidden_channels, 64)
    
    def test_forward_pass(self):
        """Test forward pass"""
        # Create dummy data
        num_nodes = 10
        num_edges = 20
        
        x = torch.randn(num_nodes, 768)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Forward pass
        out = self.model(x, edge_index, batch)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 2))  # [batch_size, num_classes]
    
    def test_attention_weights(self):
        """Test attention weight extraction"""
        num_nodes = 10
        num_edges = 20
        
        x = torch.randn(num_nodes, 768)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Forward pass with attention weights
        out, attn_weights = self.model(
            x, edge_index, batch, return_attention_weights=True
        )
        
        self.assertIsNotNone(attn_weights)
        self.assertGreater(len(attn_weights), 0)


if __name__ == '__main__':
    unittest.main()
