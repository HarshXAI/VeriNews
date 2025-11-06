"""
Data preprocessing utilities
"""

import re
from typing import List, Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        """
        Initialize preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, special characters, etc.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text)
        text = re.sub(r'#', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Remove single characters
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        return ' '.join(tokens)


class UserFeatureExtractor:
    """Extract features from user profiles"""
    
    @staticmethod
    def extract_basic_features(user_data: dict) -> dict:
        """
        Extract basic user features
        
        Args:
            user_data: User profile dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic counts
        features['followers_count'] = user_data.get('followers_count', 0)
        features['friends_count'] = user_data.get('friends_count', 0)
        features['statuses_count'] = user_data.get('statuses_count', 0)
        features['favourites_count'] = user_data.get('favourites_count', 0)
        features['listed_count'] = user_data.get('listed_count', 0)
        
        # Verification
        features['verified'] = int(user_data.get('verified', False))
        
        # Computed features
        followers = features['followers_count']
        friends = features['friends_count']
        
        # Follower-friend ratio
        if friends > 0:
            features['follower_friend_ratio'] = followers / friends
        else:
            features['follower_friend_ratio'] = followers
        
        # Engagement rate
        if followers > 0:
            features['engagement_rate'] = features['statuses_count'] / followers
        else:
            features['engagement_rate'] = 0
        
        return features
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Normalize numerical features using log transformation
        
        Args:
            df: DataFrame with features
            columns: List of columns to normalize
            
        Returns:
            DataFrame with normalized features
        """
        import numpy as np
        
        for col in columns:
            if col in df.columns:
                # Add 1 to avoid log(0)
                df[f'{col}_log'] = np.log1p(df[col])
        
        return df


def preprocess_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess news content data
    
    Args:
        news_df: DataFrame with news content
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = TextPreprocessor()
    
    # Preprocess text fields
    if 'title' in news_df.columns:
        news_df['title_clean'] = news_df['title'].apply(preprocessor.preprocess)
    
    if 'text' in news_df.columns:
        news_df['text_clean'] = news_df['text'].apply(preprocessor.preprocess)
    
    # Encode labels
    news_df['label_encoded'] = news_df['label'].map({'fake': 0, 'real': 1})
    
    return news_df


def preprocess_social_data(social_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess social context data
    
    Args:
        social_df: DataFrame with social posts
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = TextPreprocessor()
    extractor = UserFeatureExtractor()
    
    # Preprocess text
    if 'text' in social_df.columns:
        social_df['text_clean'] = social_df['text'].apply(preprocessor.preprocess)
    
    # Extract user features
    if 'user' in social_df.columns:
        user_features = social_df['user'].apply(
            lambda x: extractor.extract_basic_features(x) if isinstance(x, dict) else {}
        )
        user_features_df = pd.DataFrame(user_features.tolist())
        social_df = pd.concat([social_df, user_features_df], axis=1)
    
    return social_df


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    text = "Check out this BREAKING NEWS!!! https://fakenews.com @user #fake"
    print("Original:", text)
    print("Cleaned:", preprocessor.preprocess(text))
