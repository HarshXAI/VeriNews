"""
Feature extraction and embedding utilities
"""

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEmbedder:
    """Generate text embeddings using transformer models"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize text embedder
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (cuda, mps, or cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding array [embedding_dim]
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding


class UserFeatureEncoder:
    """Encode user features into numerical vectors"""
    
    def __init__(self):
        """Initialize user feature encoder"""
        self.feature_names = [
            'followers_count_log',
            'friends_count_log',
            'statuses_count_log',
            'favourites_count_log',
            'listed_count_log',
            'verified',
            'follower_friend_ratio',
            'engagement_rate'
        ]
    
    def encode(self, user_features_dict: dict) -> np.ndarray:
        """
        Encode user features into a vector
        
        Args:
            user_features_dict: Dictionary of user features
            
        Returns:
            Feature vector
        """
        features = []
        
        for feature_name in self.feature_names:
            value = user_features_dict.get(feature_name, 0.0)
            features.append(float(value))
        
        return np.array(features, dtype=np.float32)
    
    def encode_batch(self, user_features_list: List[dict]) -> np.ndarray:
        """
        Encode a batch of user features
        
        Args:
            user_features_list: List of user feature dictionaries
            
        Returns:
            Feature matrix [num_users, feature_dim]
        """
        features = []
        
        for user_features in tqdm(user_features_list, desc="Encoding user features"):
            features.append(self.encode(user_features))
        
        return np.vstack(features)


class FeatureCombiner:
    """Combine different types of features"""
    
    @staticmethod
    def combine_features(
        text_features: np.ndarray,
        user_features: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Combine text and user features
        
        Args:
            text_features: Text embeddings [num_samples, text_dim]
            user_features: User features [num_samples, user_dim]
            normalize: Whether to normalize features
            
        Returns:
            Combined features [num_samples, text_dim + user_dim]
        """
        # Ensure same number of samples
        assert text_features.shape[0] == user_features.shape[0], \
            "Text and user features must have same number of samples"
        
        # Normalize if requested
        if normalize:
            from sklearn.preprocessing import StandardScaler
            
            scaler_text = StandardScaler()
            scaler_user = StandardScaler()
            
            text_features = scaler_text.fit_transform(text_features)
            user_features = scaler_user.fit_transform(user_features)
        
        # Concatenate features
        combined = np.concatenate([text_features, user_features], axis=1)
        
        return combined
    
    @staticmethod
    def to_torch(features: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """
        Convert numpy features to PyTorch tensors
        
        Args:
            features: Numpy array
            device: Target device
            
        Returns:
            PyTorch tensor
        """
        return torch.from_numpy(features).float().to(device)


if __name__ == "__main__":
    # Example usage
    embedder = TextEmbedder()
    texts = ["This is fake news!", "This is real news."]
    embeddings = embedder.embed_texts(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
