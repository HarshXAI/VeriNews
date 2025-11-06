"""
Data loading utilities for FakeNewsNet dataset (CSV format)
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


class FakeNewsNetLoader:
    """Loader for FakeNewsNet dataset in CSV format"""
    
    def __init__(self, data_dir: str = "data/raw/fakenewsnet/dataset"):
        """
        Initialize the loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.sources = ["politifact", "gossipcop"]
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all news data from CSV files
        
        Returns:
            Tuple of (news_df, social_df)
            - news_df: DataFrame with news articles and metadata
            - social_df: DataFrame with tweet-to-news mappings
        """
        print("Loading FakeNewsNet dataset from CSV files...")
        
        all_news = []
        
        for source in self.sources:
            # Load fake news
            fake_file = self.data_dir / f"{source}_fake.csv"
            if fake_file.exists():
                df_fake = pd.read_csv(fake_file)
                df_fake['label'] = 'fake'
                df_fake['source'] = source
                all_news.append(df_fake)
                print(f"  ✓ Loaded {len(df_fake)} fake news from {source}")
            else:
                print(f"  ✗ File not found: {fake_file}")
            
            # Load real news
            real_file = self.data_dir / f"{source}_real.csv"
            if real_file.exists():
                df_real = pd.read_csv(real_file)
                df_real['label'] = 'real'
                df_real['source'] = source
                all_news.append(df_real)
                print(f"  ✓ Loaded {len(df_real)} real news from {source}")
            else:
                print(f"  ✗ File not found: {real_file}")
        
        if not all_news:
            raise ValueError(f"No data found in {self.data_dir}")
        
        # Combine all data
        news_df = pd.concat(all_news, ignore_index=True)
        
        # Process tweet IDs into a list
        news_df['tweet_ids'] = news_df['tweet_ids'].apply(
            lambda x: str(x).split('\t') if pd.notna(x) else []
        )
        news_df['num_tweets'] = news_df['tweet_ids'].apply(len)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  - Total articles: {len(news_df)}")
        print(f"  - Fake news: {(news_df['label'] == 'fake').sum()}")
        print(f"  - Real news: {(news_df['label'] == 'real').sum()}")
        print(f"  - Total tweets: {news_df['num_tweets'].sum():,}")
        print(f"  - Avg tweets per article: {news_df['num_tweets'].mean():.1f}")
        
        # Create social data (tweet-to-news mappings)
        social_data = []
        print("\nCreating social context mappings...")
        for idx, row in news_df.iterrows():
            for tweet_id in row['tweet_ids']:
                if tweet_id:  # Skip empty strings
                    social_data.append({
                        'tweet_id': tweet_id,
                        'news_id': row['id'],
                        'label': row['label'],
                        'source': row['source']
                    })
        
        social_df = pd.DataFrame(social_data) if social_data else pd.DataFrame()
        
        if not social_df.empty:
            print(f"  ✓ Created {len(social_df):,} tweet mappings")
        
        return news_df, social_df
    
    def load_news_only(self) -> pd.DataFrame:
        """
        Load only the news data (without social context)
        
        Returns:
            DataFrame with news articles
        """
        news_df, _ = self.load_all_data()
        return news_df


if __name__ == "__main__":
    # Test the loader
    loader = FakeNewsNetLoader()
    news_df, social_df = loader.load_all_data()
    
    print("\n" + "="*70)
    print("News DataFrame Sample:")
    print("="*70)
    print(news_df.head())
    
    print("\n" + "="*70)
    print("Social DataFrame Sample:")
    print("="*70)
    print(social_df.head())
    
    print("\n✅ Data loaded successfully!")
