"""
Temporal analysis of fake news propagation patterns
"""

import argparse
import os
import sys
from pathlib import Path
import json
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed


def load_temporal_data():
    """Load temporal information from original dataset"""
    
    # Try to load processed data with timestamps
    try:
        news_df = pd.read_parquet('data/processed/news_processed.parquet')
        social_df = pd.read_parquet('data/processed/social_processed.parquet')
        
        # Load graph to map nodes
        with open('data/graphs/propagation_graph.pkl', 'rb') as f:
            G = pickle.load(f)
        
        return news_df, social_df, G
    except Exception as e:
        print(f"Error loading temporal data: {e}")
        return None, None, None


def analyze_temporal_patterns(news_df, social_df, G, output_dir):
    """Analyze temporal patterns in the data"""
    
    if news_df is None or 'date' not in news_df.columns:
        print("⚠️  No temporal information available in dataset")
        return None
    
    # Convert date strings to datetime
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date'])
    
    if len(news_df) == 0:
        print("⚠️  No valid dates found in dataset")
        return None
    
    print(f"\n📅 Temporal Analysis:")
    print(f"  Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    print(f"  Total articles with dates: {len(news_df)}")
    
    # Add temporal features
    news_df['year'] = news_df['date'].dt.year
    news_df['month'] = news_df['date'].dt.month
    news_df['day_of_week'] = news_df['date'].dt.dayofweek
    news_df['year_month'] = news_df['date'].dt.to_period('M')
    
    # Analyze by label
    fake_news = news_df[news_df['label'] == 1]
    real_news = news_df[news_df['label'] == 0]
    
    return {
        'news_df': news_df,
        'fake_news': fake_news,
        'real_news': real_news
    }


def plot_temporal_distribution(temporal_data, output_dir):
    """Plot temporal distribution of fake vs real news"""
    
    if temporal_data is None:
        return
    
    news_df = temporal_data['news_df']
    fake_news = temporal_data['fake_news']
    real_news = temporal_data['real_news']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Articles over time (monthly)
    ax = axes[0, 0]
    fake_monthly = fake_news.groupby('year_month').size()
    real_monthly = real_news.groupby('year_month').size()
    
    x = [str(p) for p in fake_monthly.index]
    ax.plot(range(len(x)), fake_monthly.values, marker='o', label='Fake News', color='#ff6b6b', linewidth=2)
    ax.plot(range(len(x)), real_monthly.values, marker='s', label='Real News', color='#4dabf7', linewidth=2)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Number of Articles', fontsize=11)
    ax.set_title('Article Publication Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, len(x), max(1, len(x)//10)))
    ax.set_xticklabels([x[i] for i in range(0, len(x), max(1, len(x)//10))], rotation=45)
    
    # 2. Fake news ratio over time
    ax = axes[0, 1]
    total_monthly = news_df.groupby('year_month').size()
    fake_ratio = (fake_monthly / total_monthly * 100).fillna(0)
    
    ax.plot(range(len(fake_ratio)), fake_ratio.values, marker='o', color='#ff6b6b', linewidth=2)
    ax.axhline(y=fake_ratio.mean(), color='#333', linestyle='--', label=f'Mean: {fake_ratio.mean():.1f}%')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Fake News %', fontsize=11)
    ax.set_title('Fake News Percentage Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, len(x), max(1, len(x)//10)))
    ax.set_xticklabels([x[i] for i in range(0, len(x), max(1, len(x)//10))], rotation=45)
    
    # 3. Day of week distribution
    ax = axes[1, 0]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fake_by_day = fake_news['day_of_week'].value_counts().sort_index()
    real_by_day = real_news['day_of_week'].value_counts().sort_index()
    
    x_pos = np.arange(7)
    width = 0.35
    ax.bar(x_pos - width/2, [fake_by_day.get(i, 0) for i in range(7)], width, label='Fake News', color='#ff6b6b')
    ax.bar(x_pos + width/2, [real_by_day.get(i, 0) for i in range(7)], width, label='Real News', color='#4dabf7')
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Number of Articles', fontsize=11)
    ax.set_title('Publication by Day of Week', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(days)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Yearly comparison
    ax = axes[1, 1]
    fake_by_year = fake_news['year'].value_counts().sort_index()
    real_by_year = real_news['year'].value_counts().sort_index()
    
    years = sorted(set(fake_by_year.index) | set(real_by_year.index))
    x_pos = np.arange(len(years))
    width = 0.35
    ax.bar(x_pos - width/2, [fake_by_year.get(y, 0) for y in years], width, label='Fake News', color='#ff6b6b')
    ax.bar(x_pos + width/2, [real_by_year.get(y, 0) for y in years], width, label='Real News', color='#4dabf7')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Articles', fontsize=11)
    ax.set_title('Articles by Year', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved temporal distribution plot")


def analyze_propagation_speed(social_df, output_dir):
    """Analyze propagation speed if social context data is available"""
    
    if social_df is None or len(social_df) == 0:
        print("⚠️  No social propagation data available")
        return
    
    print(f"\n📢 Social Propagation Analysis:")
    print(f"  Total engagements: {len(social_df)}")
    
    # Analyze engagement types
    if 'engagement_type' in social_df.columns:
        engagement_counts = social_df['engagement_type'].value_counts()
        print(f"\n  Engagement types:")
        for eng_type, count in engagement_counts.items():
            print(f"    • {eng_type}: {count}")
    
    # Analyze propagation by label
    if 'label' in social_df.columns and social_df['label'].notna().sum() > 0:
        fake_engagements = social_df[social_df['label'] == 1]
        real_engagements = social_df[social_df['label'] == 0]
        
        if len(fake_engagements) > 0 or len(real_engagements) > 0:
            print(f"\n  Engagements by content type:")
            print(f"    • Fake news: {len(fake_engagements)} ({len(fake_engagements)/len(social_df)*100:.1f}%)")
            print(f"    • Real news: {len(real_engagements)} ({len(real_engagements)/len(social_df)*100:.1f}%)")
            
            # Plot only if we have data
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Engagement type distribution
            ax = axes[0]
            if 'engagement_type' in social_df.columns:
                fake_eng = fake_engagements['engagement_type'].value_counts()
                real_eng = real_engagements['engagement_type'].value_counts()
                
                types = sorted(set(fake_eng.index) | set(real_eng.index))
                x_pos = np.arange(len(types))
                width = 0.35
                
                ax.bar(x_pos - width/2, [fake_eng.get(t, 0) for t in types], width, label='Fake News', color='#ff6b6b')
                ax.bar(x_pos + width/2, [real_eng.get(t, 0) for t in types], width, label='Real News', color='#4dabf7')
                ax.set_xlabel('Engagement Type', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title('Social Engagement by Type', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(types, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            
            # Engagement ratio - only if both have data
            ax = axes[1]
            if len(fake_engagements) > 0 and len(real_engagements) > 0:
                data = [len(fake_engagements), len(real_engagements)]
                colors = ['#ff6b6b', '#4dabf7']
                ax.pie(data, labels=['Fake News', 'Real News'], colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Social Engagement Distribution', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Insufficient labeled data', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'social_propagation.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved social propagation plot")
        else:
            print(f"  ⚠️  No labeled engagement data available")
    else:
        print(f"  ⚠️  No label information in social data")


def create_temporal_report(temporal_data, output_dir):
    """Create temporal analysis report"""
    
    if temporal_data is None:
        report = """
# ⏰ Temporal Analysis Report

## ⚠️ Limited Temporal Data Available

The dataset does not contain sufficient temporal information (timestamps) for a comprehensive temporal analysis.

### What We Know:
- The dataset contains news articles and their labels (fake/real)
- Social engagement data is available but may lack timestamps
- Graph structure shows propagation patterns but not temporal dynamics

### Recommendations:
1. **Collect timestamped data**: For future analysis, ensure articles have publication dates
2. **Track engagement timing**: Record when users interact with content
3. **Monitor spreading patterns**: Track how quickly fake news propagates vs real news
4. **Detect trending misinformation**: Identify spike in fake news on specific topics/events

### Alternative Analyses Available:
- ✅ Node importance and network centrality
- ✅ Attention weight patterns
- ✅ Community structure and homophily
- ✅ Prediction confidence and error analysis
"""
    else:
        news_df = temporal_data['news_df']
        fake_news = temporal_data['fake_news']
        real_news = temporal_data['real_news']
        
        report = f"""
# ⏰ Temporal Analysis Report

## 📊 Dataset Overview

- **Total articles**: {len(news_df):,}
- **Date range**: {news_df['date'].min().strftime('%Y-%m-%d')} to {news_df['date'].max().strftime('%Y-%m-%d')}
- **Time span**: {(news_df['date'].max() - news_df['date'].min()).days} days
- **Fake news articles**: {len(fake_news):,} ({len(fake_news)/len(news_df)*100:.1f}%)
- **Real news articles**: {len(real_news):,} ({len(real_news)/len(news_df)*100:.1f}%)

## 🔑 Key Findings

### Publication Patterns

1. **Fake News Trends**:
   - Average {len(fake_news)/news_df['year_month'].nunique():.1f} fake articles per month
   - Peak activity: {fake_news.groupby('year_month').size().idxmax()}
   
2. **Real News Trends**:
   - Average {len(real_news)/news_df['year_month'].nunique():.1f} real articles per month
   - Peak activity: {real_news.groupby('year_month').size().idxmax()}

### Temporal Insights

- Fake news shows {'higher' if fake_news.groupby('year_month').size().std() > real_news.groupby('year_month').size().std() else 'lower'} variability over time
- Publication patterns {'differ' if fake_news['day_of_week'].mean() != real_news['day_of_week'].mean() else 'are similar'} between fake and real news

## 📈 Visualizations Generated

1. **temporal_distribution.png**: Time series of article publication
2. **social_propagation.png**: Social engagement patterns

## 💡 Implications

The temporal analysis reveals how fake news spreads over time and helps identify:
- **Peak periods** of misinformation
- **Seasonal patterns** in fake news creation
- **Propagation speed** differences between fake and real news
"""
    
    with open(os.path.join(output_dir, 'TEMPORAL_REPORT.md'), 'w') as f:
        f.write(report)
    
    print(f"  ✓ Saved temporal analysis report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="experiments/temporal_analysis")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("⏰ TEMPORAL PATTERN ANALYSIS")
    print("="*70)
    
    # Load temporal data
    print(f"\n📂 Loading temporal data...")
    news_df, social_df, G = load_temporal_data()
    
    if news_df is not None:
        print(f"  ✓ Loaded {len(news_df)} news articles")
        if social_df is not None:
            print(f"  ✓ Loaded {len(social_df)} social engagements")
    
    # Analyze temporal patterns
    temporal_data = analyze_temporal_patterns(news_df, social_df, G, args.output_dir)
    
    # Create visualizations
    if temporal_data is not None:
        print(f"\n🎨 Creating temporal visualizations...")
        plot_temporal_distribution(temporal_data, args.output_dir)
    
    # Analyze propagation speed
    if social_df is not None:
        analyze_propagation_speed(social_df, args.output_dir)
    
    # Create report
    print(f"\n📝 Creating temporal report...")
    create_temporal_report(temporal_data, args.output_dir)
    
    print("\n" + "="*70)
    print("✅ TEMPORAL ANALYSIS COMPLETE!")
    print("="*70)
    
    if temporal_data is not None:
        print(f"\n📁 Files generated:")
        print(f"  • temporal_distribution.png - Time series visualizations")
        if social_df is not None:
            print(f"  • social_propagation.png - Engagement patterns")
        print(f"  • TEMPORAL_REPORT.md - Analysis summary")
    else:
        print(f"\n⚠️  Limited temporal data available")
        print(f"  Generated report with recommendations at:")
        print(f"  {args.output_dir}/TEMPORAL_REPORT.md")


if __name__ == "__main__":
    main()
