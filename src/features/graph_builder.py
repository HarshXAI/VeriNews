"""
Graph construction utilities for building propagation networks
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx


class PropagationGraphBuilder:
    """Build propagation graphs from social media data"""
    
    def __init__(self):
        """Initialize graph builder"""
        self.graph = nx.DiGraph()
    
    def add_retweet_edges(self, retweets_df: pd.DataFrame):
        """
        Add retweet edges to the graph
        
        Args:
            retweets_df: DataFrame with retweet information
        """
        for _, row in retweets_df.iterrows():
            if 'user' in row and 'retweeted_status' in row:
                source_user = row['user'].get('id_str')
                
                retweeted = row.get('retweeted_status', {})
                if isinstance(retweeted, dict) and 'user' in retweeted:
                    target_user = retweeted['user'].get('id_str')
                    
                    if source_user and target_user:
                        self.graph.add_edge(
                            source_user,
                            target_user,
                            edge_type='retweet',
                            weight=1.0,
                            timestamp=row.get('created_at')
                        )
    
    def add_reply_edges(self, tweets_df: pd.DataFrame):
        """
        Add reply edges to the graph
        
        Args:
            tweets_df: DataFrame with tweet information
        """
        for _, row in tweets_df.iterrows():
            if 'user' in row and 'in_reply_to_user_id_str' in row:
                source_user = row['user'].get('id_str')
                target_user = row.get('in_reply_to_user_id_str')
                
                if source_user and target_user:
                    self.graph.add_edge(
                        source_user,
                        target_user,
                        edge_type='reply',
                        weight=1.0,
                        timestamp=row.get('created_at')
                    )
    
    def add_mention_edges(self, tweets_df: pd.DataFrame):
        """
        Add mention edges to the graph
        
        Args:
            tweets_df: DataFrame with tweet information
        """
        for _, row in tweets_df.iterrows():
            if 'user' in row and 'entities' in row:
                source_user = row['user'].get('id_str')
                entities = row.get('entities', {})
                
                if isinstance(entities, dict) and 'user_mentions' in entities:
                    mentions = entities['user_mentions']
                    
                    if isinstance(mentions, list):
                        for mention in mentions:
                            target_user = mention.get('id_str')
                            
                            if source_user and target_user:
                                self.graph.add_edge(
                                    source_user,
                                    target_user,
                                    edge_type='mention',
                                    weight=1.0,
                                    timestamp=row.get('created_at')
                                )
    
    def build_graph(
        self,
        tweets_df: pd.DataFrame,
        retweets_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Build complete propagation graph
        
        Args:
            tweets_df: DataFrame with tweets
            retweets_df: DataFrame with retweets
            
        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        
        # Add different types of edges
        self.add_retweet_edges(retweets_df)
        self.add_reply_edges(tweets_df)
        self.add_mention_edges(tweets_df)
        
        return self.graph
    
    def get_statistics(self) -> Dict:
        """
        Get graph statistics
        
        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
        }
        
        # Only calculate these if graph is not empty
        if stats['num_nodes'] > 0:
            stats['density'] = nx.density(self.graph)
            stats['is_connected'] = nx.is_weakly_connected(self.graph) if stats['num_edges'] > 0 else False
            stats['num_components'] = nx.number_weakly_connected_components(self.graph)
        else:
            stats['density'] = 0
            stats['is_connected'] = False
            stats['num_components'] = 0
            
        return stats


class HeterogeneousGraphBuilder:
    """Build heterogeneous graphs with multiple node and edge types"""
    
    def __init__(self):
        """Initialize heterogeneous graph builder"""
        self.data = HeteroData()
    
    def add_user_nodes(
        self,
        user_features: torch.Tensor,
        user_ids: List[str]
    ):
        """
        Add user nodes to heterogeneous graph
        
        Args:
            user_features: Tensor of user features [num_users, feature_dim]
            user_ids: List of user IDs
        """
        self.data['user'].x = user_features
        self.data['user'].node_id = user_ids
    
    def add_post_nodes(
        self,
        post_features: torch.Tensor,
        post_ids: List[str],
        labels: torch.Tensor = None
    ):
        """
        Add post nodes to heterogeneous graph
        
        Args:
            post_features: Tensor of post features [num_posts, feature_dim]
            post_ids: List of post IDs
            labels: Optional labels for posts
        """
        self.data['post'].x = post_features
        self.data['post'].node_id = post_ids
        
        if labels is not None:
            self.data['post'].y = labels
    
    def add_edges(
        self,
        edge_index: torch.Tensor,
        source_type: str,
        edge_type: str,
        target_type: str,
        edge_attr: torch.Tensor = None
    ):
        """
        Add edges to heterogeneous graph
        
        Args:
            edge_index: Edge indices [2, num_edges]
            source_type: Source node type
            edge_type: Edge type
            target_type: Target node type
            edge_attr: Optional edge attributes
        """
        self.data[source_type, edge_type, target_type].edge_index = edge_index
        
        if edge_attr is not None:
            self.data[source_type, edge_type, target_type].edge_attr = edge_attr
    
    def get_data(self) -> HeteroData:
        """
        Get the heterogeneous graph data
        
        Returns:
            HeteroData object
        """
        return self.data


def convert_to_pytorch_geometric(
    G: nx.Graph,
    node_features: Dict[str, np.ndarray],
    labels: Dict[str, int] = None
) -> torch.Tensor:
    """
    Convert NetworkX graph to PyTorch Geometric format
    
    Args:
        G: NetworkX graph
        node_features: Dictionary mapping node IDs to feature arrays
        labels: Optional dictionary mapping node IDs to labels
        
    Returns:
        PyTorch Geometric Data object
    """
    # Add features to nodes
    for node in G.nodes():
        if node in node_features:
            G.nodes[node]['x'] = node_features[node]
        
        if labels and node in labels:
            G.nodes[node]['y'] = labels[node]
    
    # Convert to PyTorch Geometric
    data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['weight'])
    
    return data


if __name__ == "__main__":
    # Example usage
    builder = PropagationGraphBuilder()
    print("Propagation graph builder initialized")
