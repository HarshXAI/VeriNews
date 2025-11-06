"""
Download FakeNewsNet dataset

Usage:
    python scripts/download_dataset.py --output data/raw/fakenewsnet
"""

import argparse
import os
import subprocess
import sys


def download_dataset(output_dir: str):
    """
    Download FakeNewsNet dataset from GitHub
    
    Args:
        output_dir: Directory to save the dataset
    """
    print("Downloading FakeNewsNet dataset...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone the repository
    repo_url = "https://github.com/KaiDMML/FakeNewsNet.git"
    
    try:
        subprocess.run(
            ["git", "clone", repo_url, output_dir],
            check=True
        )
        print(f"\n✓ Successfully downloaded FakeNewsNet to {output_dir}")
        print("\nDataset structure:")
        print("  - politifact/")
        print("  - gossipcop/")
        print("\nNext steps:")
        print("  1. Run preprocessing: python src/data/preprocess.py")
        print("  2. Build graphs: python src/features/build_graph.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  {repo_url}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n✗ Error: git is not installed")
        print("\nPlease install git or download manually from:")
        print(f"  {repo_url}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download FakeNewsNet dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw/fakenewsnet",
        help="Output directory for dataset"
    )
    
    args = parser.parse_args()
    download_dataset(args.output)


if __name__ == "__main__":
    main()
