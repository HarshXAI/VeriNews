"""
Visualization utilities for graphs and attention weights
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import List, Tuple, Optional


class GraphVisualizer:
    """Visualize propagation graphs"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_propagation_tree(
        self,
        tree: nx.DiGraph,
        root_node: int,
        node_labels: dict = None,
        save_path: Optional[str] = None
    ):
        """
        Plot propagation tree
        
        Args:
            tree: NetworkX directed graph
            root_node: Root node ID
            node_labels: Optional dictionary of node labels
            save_path: Optional path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        # Create hierarchical layout
        pos = nx.spring_layout(tree, k=2, iterations=50)
        
        # Get edge weights for coloring
        edges = tree.edges()
        weights = [tree[u][v].get('weight', 1.0) for u, v in edges]
        
        # Normalize weights for visualization
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
        
        # Draw nodes
        node_colors = ['red' if n == root_node else 'lightblue' for n in tree.nodes()]
        nx.draw_networkx_nodes(
            tree, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            tree, pos,
            width=[w * 3 for w in weights] if weights else 1,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            edge_color=weights if weights else 'black',
            edge_cmap=plt.cm.Blues
        )
        
        # Draw labels
        if node_labels:
            nx.draw_networkx_labels(tree, pos, node_labels, font_size=8)
        
        plt.title("News Propagation Tree", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved propagation tree to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        node_labels: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot attention weights as heatmap
        
        Args:
            attention_weights: Attention weight matrix
            node_labels: Optional list of node labels
            save_path: Optional path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        sns.heatmap(
            attention_weights,
            cmap='YlOrRd',
            xticklabels=node_labels if node_labels else False,
            yticklabels=node_labels if node_labels else False,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.title("Attention Weights Heatmap", fontsize=16, fontweight='bold')
        plt.xlabel("Target Nodes")
        plt.ylabel("Source Nodes")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_influential_users(
        self,
        user_scores: List[Tuple[int, float]],
        user_labels: dict = None,
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot top influential users
        
        Args:
            user_scores: List of (user_id, score) tuples
            user_labels: Optional dictionary mapping user IDs to labels
            top_k: Number of top users to plot
            save_path: Optional path to save figure
        """
        # Sort and select top k
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        users = [str(user_labels.get(u[0], f"User {u[0]}")) if user_labels 
                else f"User {u[0]}" for u in user_scores]
        scores = [u[1] for u in user_scores]
        
        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(users))
        plt.barh(y_pos, scores, color='steelblue', alpha=0.8)
        
        plt.yticks(y_pos, users)
        plt.xlabel('Influence Score', fontsize=12)
        plt.title(f'Top {top_k} Influential Users', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved influential users plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


class MetricsVisualizer:
    """Visualize model performance metrics"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Optional path to save figure
        """
        if class_names is None:
            class_names = ['Fake', 'Real']
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: List[float],
        val_metrics: List[float],
        metric_name: str = "F1 Score",
        save_path: Optional[str] = None
    ):
        """
        Plot training history
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_metrics: Training metrics
            val_metrics: Validation metrics
            metric_name: Name of the metric
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot metrics
        ax2.plot(epochs, train_metrics, 'b-o', label=f'Training {metric_name}', linewidth=2)
        ax2.plot(epochs, val_metrics, 'r-o', label=f'Validation {metric_name}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title(f'Training and Validation {metric_name}', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics: dict,
        save_path: Optional[str] = None
    ):
        """
        Plot metrics comparison bar chart
        
        Args:
            metrics: Dictionary of metrics
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create color palette
        colors = sns.color_palette("husl", len(metric_names))
        
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics comparison to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    print("Visualization module loaded")
