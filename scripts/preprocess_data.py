"""
Preprocess FakeNewsNet data

Usage:
    python scripts/preprocess_data.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import FakeNewsNetLoader
from src.data.preprocessor import preprocess_news_data, preprocess_social_data


def main():
    parser = argparse.ArgumentParser(description="Preprocess FakeNewsNet dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/fakenewsnet/dataset",
        help="Input directory with CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input directory not found: {args.input}")
        print("\nPlease download the dataset first:")
        print("  python scripts/download_dataset.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*70)
    print("PREPROCESSING FAKENEWSNET DATASET")
    print("="*70)
    
    # Load data
    loader = FakeNewsNetLoader(args.input)
    news_df, social_df = loader.load_all_data()
    
    # Preprocess news data
    print("\n" + "="*70)
    print("PREPROCESSING NEWS DATA")
    print("="*70)
    news_df = preprocess_news_data(news_df)
    
    # Preprocess social data
    print("\n" + "="*70)
    print("PREPROCESSING SOCIAL DATA")
    print("="*70)
    social_df = preprocess_social_data(social_df)
    
    # Save processed data
    print("\n" + "="*70)
    print("SAVING PROCESSED DATA")
    print("="*70)
    
    news_output = os.path.join(args.output, "news_processed.parquet")
    social_output = os.path.join(args.output, "social_processed.parquet")
    
    news_df.to_parquet(news_output)
    print(f"  ✓ Saved news data: {news_output}")
    print(f"    - Shape: {news_df.shape}")
    print(f"    - Size: {os.path.getsize(news_output) / 1024 / 1024:.2f} MB")
    
    social_df.to_parquet(social_output)
    print(f"  ✓ Saved social data: {social_output}")
    print(f"    - Shape: {social_df.shape}")
    print(f"    - Size: {os.path.getsize(social_output) / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessed data saved to: {args.output}")
    print("\nNext step: Build propagation graphs")
    print("  python scripts/build_graphs.py")


if __name__ == "__main__":
    main()
