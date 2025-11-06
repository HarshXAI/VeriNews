"""
Graph Attention Network (GAT) model for fake news detection
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class FakeNewsGAT(nn.Module):
    """
    Graph Attention Network for Fake News Detection
    
    This model uses multiple GAT layers to learn propagation patterns
    and classify news articles as fake or real.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
        concat_heads: bool = True,
        use_batch_norm: bool = True
    ):
        """
        Initialize GAT model
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Number of output classes
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat_heads: Whether to concatenate or average attention heads
            use_batch_norm: Whether to use batch normalization
        """
        super(FakeNewsGAT, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads
        self.use_batch_norm = use_batch_norm
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout
            )
        )
        
        if use_batch_norm:
            bn_dim = hidden_channels * num_heads if concat_heads else hidden_channels
            self.batch_norms.append(nn.BatchNorm1d(bn_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * num_heads if concat_heads else hidden_channels
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_channels,
                    heads=num_heads,
                    concat=concat_heads,
                    dropout=dropout
                )
            )
            
            if use_batch_norm:
                bn_dim = hidden_channels * num_heads if concat_heads else hidden_channels
                self.batch_norms.append(nn.BatchNorm1d(bn_dim))
        
        # Last layer (single head)
        in_dim = hidden_channels * num_heads if concat_heads else hidden_channels
        self.convs.append(
            GATConv(
                in_dim,
                hidden_channels,
                heads=1,
                concat=False,
                dropout=dropout
            )
        )
        
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Store attention weights for interpretability
        self.attention_weights = []
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for graph-level prediction
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Log probabilities or tuple of (log_probs, attention_weights)
        """
        self.attention_weights = []
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            # Get attention weights
            if return_attention_weights:
                x, (edge_idx, attn) = conv(
                    x, edge_index, return_attention_weights=True
                )
                self.attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)
            
            # Apply batch normalization
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        if return_attention_weights:
            return F.log_softmax(x, dim=1), self.attention_weights
        else:
            return F.log_softmax(x, dim=1)
    
    def get_attention_weights(self):
        """
        Get stored attention weights from last forward pass
        
        Returns:
            List of tuples (edge_index, attention_weights)
        """
        return self.attention_weights


class HeterogeneousGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network for multi-type nodes
    """
    
    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int,
        out_channels: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize heterogeneous GAT
        
        Args:
            in_channels_dict: Dictionary of input dimensions per node type
            hidden_channels: Hidden dimension
            out_channels: Number of output classes
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(HeterogeneousGAT, self).__init__()
        
        from torch_geometric.nn import HeteroConv
        
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        # First layer
        conv_dict = {}
        for node_type in in_channels_dict:
            conv_dict[('user', 'interacts', 'post')] = GATConv(
                (-1, -1), hidden_channels, heads=num_heads, add_self_loops=False
            )
        
        self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels * num_heads, out_channels)
        self.dropout = dropout
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass for heterogeneous graph
        
        Args:
            x_dict: Dictionary of node features per type
            edge_index_dict: Dictionary of edge indices per edge type
            
        Returns:
            Log probabilities
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                     for key, x in x_dict.items()}
        
        # Use post node features for classification
        x = x_dict['post']
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # Example usage
    model = FakeNewsGAT(
        in_channels=768,
        hidden_channels=128,
        out_channels=2,
        num_layers=3,
        num_heads=8
    )
    
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
