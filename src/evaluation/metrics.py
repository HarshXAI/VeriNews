"""
Evaluation metrics and utilities
"""

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
        }
        
        # Add AUC if probabilities provided
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list = None
    ):
        """
        Print classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
        """
        if target_names is None:
            target_names = ['Fake', 'Real']
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))


class ModelEvaluator:
    """Evaluate GAT model on test data"""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize evaluator
        
        Args:
            model: GAT model
            device: Device to use
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, test_loader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (metrics, predictions, true_labels)
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Get predictions
            pred = out.argmax(dim=1)
            prob = torch.exp(out)[:, 1]  # Probability of positive class
            
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Compute metrics
        calculator = MetricsCalculator()
        metrics = calculator.compute_metrics(y_true, y_pred, y_prob)
        
        return metrics, y_pred, y_true
    
    def evaluate_and_report(self, test_loader):
        """
        Evaluate model and print detailed report
        
        Args:
            test_loader: Test data loader
        """
        metrics, y_pred, y_true = self.evaluate(test_loader)
        
        # Print metrics
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Print confusion matrix
        calculator = MetricsCalculator()
        cm = calculator.compute_confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Fake   Real")
        print(f"Actual Fake  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Real  {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Print classification report
        calculator.print_classification_report(y_true, y_pred)
        
        return metrics


if __name__ == "__main__":
    print("Evaluation module loaded")
