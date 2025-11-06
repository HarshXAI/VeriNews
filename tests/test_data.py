"""
Unit tests for data loading
"""

import unittest
from src.data import TextPreprocessor, UserFeatureExtractor


class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "Check out https://example.com @user #news!!!"
        cleaned = self.preprocessor.clean_text(text)
        
        # Should remove URL, mention, and special chars
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("@user", cleaned)
    
    def test_tokenize(self):
        """Test tokenization"""
        text = "This is a test sentence"
        tokens = self.preprocessor.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_preprocess(self):
        """Test full preprocessing pipeline"""
        text = "Breaking NEWS! Visit https://fake.com"
        processed = self.preprocessor.preprocess(text)
        
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)


class TestUserFeatureExtractor(unittest.TestCase):
    """Test user feature extraction"""
    
    def setUp(self):
        self.extractor = UserFeatureExtractor()
    
    def test_extract_basic_features(self):
        """Test basic feature extraction"""
        user_data = {
            'followers_count': 1000,
            'friends_count': 500,
            'statuses_count': 2000,
            'verified': True
        }
        
        features = self.extractor.extract_basic_features(user_data)
        
        self.assertIn('followers_count', features)
        self.assertIn('follower_friend_ratio', features)
        self.assertEqual(features['verified'], 1)


if __name__ == '__main__':
    unittest.main()
