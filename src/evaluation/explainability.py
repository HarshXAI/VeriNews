"""
Explainability and interpretability utilities
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import networkx as nx


class AttentionAnalyzer:
    """Analyze and visualize attention weights from GAT"""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize analyzer
        
        Args:
            model: GAT model
            device: Device to use
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None
    ) -> List[Tuple]:
        """
        Get attention weights from model
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            
        Returns:
            List of (edge_index, attention_weights) tuples
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        if batch is not None:
            batch = batch.to(self.device)
        
        # Forward pass with attention weights
        _, attention_weights = self.model(
            x, edge_index, batch, return_attention_weights=True
        )
        
        return attention_weights
    
    def identify_influential_users(
        self,
        attention_weights: List[Tuple],
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Identify most influential users based on attention weights
        
        Args:
            attention_weights: List of (edge_index, weights) tuples
            top_k: Number of top users to return
            
        Returns:
            List of (user_id, influence_score) tuples
        """
        # Aggregate attention weights by target node
        node_scores = {}
        
        for edge_idx, attn in attention_weights:
            edge_idx = edge_idx.cpu().numpy()
            attn = attn.cpu().numpy()
            
            # Sum attention weights for each target node
            for i in range(edge_idx.shape[1]):
                target_node = edge_idx[1, i]
                
                if target_node not in node_scores:
                    node_scores[target_node] = 0.0
                
                node_scores[target_node] += attn[i].mean()
        
        # Sort by influence score
        influential_users = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return influential_users
    
    def build_propagation_tree(
        self,
        edge_index: torch.Tensor,
        attention_weights: torch.Tensor,
        root_node: int
    ) -> nx.DiGraph:
        """
        Build propagation tree from attention weights
        
        Args:
            edge_index: Edge indices
            attention_weights: Attention weights for edges
            root_node: Root node of the tree
            
        Returns:
            NetworkX directed graph representing propagation tree
        """
        G = nx.DiGraph()
        
        edge_idx = edge_index.cpu().numpy()
        attn = attention_weights.cpu().numpy()
        
        # Add edges with attention weights
        for i in range(edge_idx.shape[1]):
            source = edge_idx[0, i]
            target = edge_idx[1, i]
            weight = attn[i].mean()
            
            G.add_edge(source, target, weight=weight)
        
        # Extract tree rooted at root_node
        if root_node in G:
            tree_edges = nx.bfs_edges(G, root_node)
            tree = G.edge_subgraph(tree_edges)
        else:
            tree = nx.DiGraph()
        
        return tree
    
    def compute_propagation_metrics(
        self,
        tree: nx.DiGraph
    ) -> Dict[str, float]:
        """
        Compute propagation metrics for a tree
        
        Args:
            tree: Propagation tree
            
        Returns:
            Dictionary of metrics
        """
        if len(tree) == 0:
            return {
                'depth': 0,
                'breadth': 0,
                'viral_coefficient': 0.0
            }
        
        # Compute depth (longest path from root)
        try:
            depth = nx.dag_longest_path_length(tree)
        except:
            depth = 0
        
        # Compute breadth (number of nodes)
        breadth = tree.number_of_nodes()
        
        # Compute viral coefficient (average out-degree)
        if breadth > 0:
            viral_coefficient = tree.number_of_edges() / breadth
        else:
            viral_coefficient = 0.0
        
        return {
            'depth': depth,
            'breadth': breadth,
            'viral_coefficient': viral_coefficient
        }


class FeatureImportanceAnalyzer:
    """Analyze feature importance"""
    
    @staticmethod
    def compute_feature_importance(
        model: torch.nn.Module,
        data_loader,
        feature_names: List[str],
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Compute feature importance using gradient-based attribution
        
        Args:
            model: GAT model
            data_loader: Data loader
            feature_names: List of feature names
            device: Device to use
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        model.to(device)
        model.eval()
        
        feature_gradients = []
        
        for batch in data_loader:
            batch = batch.to(device)
            batch.x.requires_grad = True
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Backward pass
            loss = out.sum()
            loss.backward()
            
            # Get gradients
            if batch.x.grad is not None:
                feature_gradients.append(batch.x.grad.abs().mean(dim=0).cpu().numpy())
        
        # Average gradients
        avg_gradients = np.mean(feature_gradients, axis=0)
        
        # Create importance dictionary
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(avg_gradients):
                importance[name] = float(avg_gradients[i])
        
        return importance


if __name__ == "__main__":
    print("Explainability module loaded")
