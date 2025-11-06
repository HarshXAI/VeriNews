"""
Week 3-4: Enhanced feature engineering
Extract sentiment, source credibility, NER, readability, and style features
"""

import argparse
import os
import sys
from pathlib import Path
import pickle

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_sentiment_features(texts):
    """Extract sentiment scores"""
    try:
        from transformers import pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        
        print("  Extracting sentiment features...")
        sentiments = []
        for text in tqdm(texts, desc="Sentiment"):
            try:
                result = sentiment_analyzer(text[:512])[0]
                score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                sentiments.append(score)
            except:
                sentiments.append(0.0)
        
        return np.array(sentiments).reshape(-1, 1)
    except Exception as e:
        print(f"  ⚠️  Sentiment analysis failed: {e}")
        return np.zeros((len(texts), 1))


def extract_source_features(news_df):
    """Calculate source credibility scores"""
    print("  Calculating source credibility...")
    
    if 'source' not in news_df.columns:
        return np.zeros((len(news_df), 3))
    
    # Group by source
    source_stats = news_df.groupby('source').agg({
        'label': ['mean', 'count', 'std']
    }).fillna(0)
    
    source_stats.columns = ['fake_ratio', 'article_count', 'label_std']
    
    # Add credibility score (inverse of fake ratio)
    source_stats['credibility'] = 1 - source_stats['fake_ratio']
    
    # Map back to articles
    features = []
    for _, row in news_df.iterrows():
        source = row.get('source', 'unknown')
        if source in source_stats.index:
            stats = source_stats.loc[source]
            features.append([
                stats['credibility'],
                np.log1p(stats['article_count']),
                stats['label_std']
            ])
        else:
            features.append([0.5, 0.0, 0.0])
    
    return np.array(features)


def extract_ner_features(texts):
    """Extract named entity counts"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        print("  Extracting named entities...")
        features = []
        for text in tqdm(texts, desc="NER"):
            try:
                doc = nlp(text[:1000])  # Limit length
                features.append([
                    len([e for e in doc.ents if e.label_ == 'PERSON']),
                    len([e for e in doc.ents if e.label_ == 'ORG']),
                    len([e for e in doc.ents if e.label_ == 'GPE']),
                    len([e for e in doc.ents if e.label_ == 'DATE']),
                    len(doc.ents)  # Total entities
                ])
            except:
                features.append([0, 0, 0, 0, 0])
        
        return np.array(features)
    except Exception as e:
        print(f"  ⚠️  NER failed: {e}")
        print(f"  Install spaCy model: python -m spacy download en_core_web_sm")
        return np.zeros((len(texts), 5))


def extract_readability_features(texts):
    """Extract readability scores"""
    try:
        import textstat
        
        print("  Calculating readability scores...")
        features = []
        for text in tqdm(texts, desc="Readability"):
            try:
                features.append([
                    textstat.flesch_reading_ease(text),
                    textstat.flesch_kincaid_grade(text),
                    textstat.gunning_fog(text),
                    textstat.automated_readability_index(text),
                    textstat.coleman_liau_index(text)
                ])
            except:
                features.append([0, 0, 0, 0, 0])
        
        return np.array(features)
    except Exception as e:
        print(f"  ⚠️  Readability analysis failed: {e}")
        print(f"  Install textstat: pip install textstat")
        return np.zeros((len(texts), 5))


def extract_style_features(texts):
    """Extract writing style features"""
    print("  Extracting writing style features...")
    
    features = []
    for text in tqdm(texts, desc="Style"):
        try:
            # Punctuation features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
            
            # Sentence features
            sentences = text.split('.')
            avg_sentence_len = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Word features
            words = text.split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            unique_words = len(set(words)) / (len(words) + 1)
            
            # Emotional words (simple heuristic)
            emotional_words = ['amazing', 'terrible', 'shocking', 'unbelievable', 'breaking', 'urgent']
            emotional_ratio = sum(1 for w in words if w.lower() in emotional_words) / (len(words) + 1)
            
            features.append([
                exclamation_count,
                question_count,
                caps_ratio,
                avg_sentence_len,
                avg_word_len,
                unique_words,
                emotional_ratio
            ])
        except:
            features.append([0, 0, 0, 0, 0, 0, 0])
    
    return np.array(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-path", type=str, default="data/processed/news_processed.parquet")
    parser.add_argument("--graph-path", type=str, default="data/graphs/graph_data.pt")
    parser.add_argument("--output-path", type=str, default="data/graphs/enhanced_features.pt")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎨 ENHANCED FEATURE ENGINEERING")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data...")
    news_df = pd.read_parquet(args.news_path)
    graph_data = torch.load(args.graph_path, weights_only=False)
    
    if args.limit:
        news_df = news_df.head(args.limit)
    
    print(f"  ✓ Loaded {len(news_df)} articles")
    print(f"  ✓ Current features: {graph_data.x.shape}")
    
    # Get texts
    texts = news_df['text'].fillna('').tolist()
    
    # Extract all features
    print(f"\n🔍 Extracting enhanced features...")
    
    all_features = [graph_data.x.cpu().numpy()]  # Start with existing features
    
    # 1. Sentiment
    sentiment_feats = extract_sentiment_features(texts)
    all_features.append(sentiment_feats)
    print(f"  ✓ Sentiment: {sentiment_feats.shape}")
    
    # 2. Source credibility
    source_feats = extract_source_features(news_df)
    all_features.append(source_feats)
    print(f"  ✓ Source: {source_feats.shape}")
    
    # 3. Named entities
    ner_feats = extract_ner_features(texts)
    all_features.append(ner_feats)
    print(f"  ✓ NER: {ner_feats.shape}")
    
    # 4. Readability
    readability_feats = extract_readability_features(texts)
    all_features.append(readability_feats)
    print(f"  ✓ Readability: {readability_feats.shape}")
    
    # 5. Writing style
    style_feats = extract_style_features(texts)
    all_features.append(style_feats)
    print(f"  ✓ Style: {style_feats.shape}")
    
    # Combine all features
    print(f"\n🔗 Combining features...")
    combined_features = np.hstack(all_features)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)
    
    # Convert to torch
    enhanced_x = torch.FloatTensor(combined_features)
    
    print(f"  ✓ Original features: {graph_data.x.shape}")
    print(f"  ✓ Enhanced features: {enhanced_x.shape}")
    print(f"  ✓ Added {enhanced_x.shape[1] - graph_data.x.shape[1]} new features")
    
    # Update graph data
    graph_data.x = enhanced_x
    
    # Save
    print(f"\n💾 Saving enhanced features...")
    torch.save(graph_data, args.output_path)
    print(f"  ✓ Saved to: {args.output_path}")
    
    # Save feature names for reference
    feature_names = {
        'original': list(range(384)),
        'sentiment': ['sentiment_score'],
        'source': ['source_credibility', 'source_article_count', 'source_label_std'],
        'ner': ['num_persons', 'num_orgs', 'num_locations', 'num_dates', 'num_entities_total'],
        'readability': ['flesch_ease', 'flesch_kincaid', 'gunning_fog', 'ari', 'coleman_liau'],
        'style': ['exclamations', 'questions', 'caps_ratio', 'avg_sent_len', 'avg_word_len', 'lexical_diversity', 'emotional_ratio']
    }
    
    with open(args.output_path.replace('.pt', '_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\nFeature breakdown:")
    print(f"  • Original (BERT): 384 features")
    print(f"  • Sentiment: 1 feature")
    print(f"  • Source credibility: 3 features")
    print(f"  • Named entities: 5 features")
    print(f"  • Readability: 5 features")
    print(f"  • Writing style: 7 features")
    print(f"  • TOTAL: {enhanced_x.shape[1]} features")
    
    print(f"\nNext step: Train with enhanced features:")
    print(f"  python scripts/train_gat_large.py --data-path {args.output_path}")


if __name__ == "__main__":
    main()
