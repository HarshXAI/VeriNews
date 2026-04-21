"""
Clean Feature Engineering v2 - No Label Leakage
Extracts sentiment, readability, NER, and writing style features.
Source features use only article count (no label information).
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
    """Extract sentiment scores using DistilBERT"""
    try:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            batch_size=64
        )

        print("  Extracting sentiment features...")
        sentiments = []
        # Process in batches for speed
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
            batch = [t[:512] if t else "empty" for t in texts[i:i+batch_size]]
            try:
                results = sentiment_analyzer(batch)
                for r in results:
                    score = r['score'] if r['label'] == 'POSITIVE' else -r['score']
                    sentiments.append(score)
            except Exception:
                sentiments.extend([0.0] * len(batch))

        return np.array(sentiments).reshape(-1, 1)
    except Exception as e:
        print(f"  Warning: Sentiment analysis failed: {e}")
        return np.zeros((len(texts), 1))


def extract_source_features(news_df):
    """Extract source features WITHOUT label leakage.
    Only uses: article count per source, source name length (proxy for credibility).
    """
    print("  Calculating source features (no label leakage)...")

    if 'source' not in news_df.columns:
        return np.zeros((len(news_df), 2))

    # Only use label-free statistics
    source_counts = news_df.groupby('source').size().rename('article_count')

    features = []
    for _, row in news_df.iterrows():
        source = row.get('source', 'unknown')
        if source in source_counts.index:
            count = source_counts[source]
            features.append([
                np.log1p(count),           # log article count
                len(str(source)),          # source name length (heuristic)
            ])
        else:
            features.append([0.0, 0.0])

    return np.array(features)


def extract_ner_features(texts):
    """Extract named entity counts"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

        print("  Extracting named entities...")
        features = []
        # Process in batches using nlp.pipe
        batch_size = 256
        for doc in tqdm(nlp.pipe(
            [t[:1000] if t else "" for t in texts],
            batch_size=batch_size,
            n_process=1
        ), total=len(texts), desc="NER"):
            try:
                ent_counts = {}
                for e in doc.ents:
                    ent_counts[e.label_] = ent_counts.get(e.label_, 0) + 1
                features.append([
                    ent_counts.get('PERSON', 0),
                    ent_counts.get('ORG', 0),
                    ent_counts.get('GPE', 0),
                    ent_counts.get('DATE', 0),
                    len(doc.ents)
                ])
            except Exception:
                features.append([0, 0, 0, 0, 0])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"  Warning: NER failed: {e}")
        return np.zeros((len(texts), 5), dtype=np.float32)


def extract_readability_features(texts):
    """Extract readability scores"""
    try:
        import textstat

        print("  Calculating readability scores...")
        features = []
        for text in tqdm(texts, desc="Readability"):
            try:
                if len(text.strip()) < 10:
                    features.append([0, 0, 0, 0, 0])
                else:
                    features.append([
                        textstat.flesch_reading_ease(text),
                        textstat.flesch_kincaid_grade(text),
                        textstat.gunning_fog(text),
                        textstat.automated_readability_index(text),
                        textstat.coleman_liau_index(text)
                    ])
            except Exception:
                features.append([0, 0, 0, 0, 0])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"  Warning: Readability failed: {e}")
        return np.zeros((len(texts), 5), dtype=np.float32)


def extract_style_features(texts):
    """Extract writing style features"""
    print("  Extracting writing style features...")

    features = []
    for text in tqdm(texts, desc="Style"):
        try:
            if not text or len(text.strip()) < 5:
                features.append([0.0] * 7)
                continue

            words = text.split()
            n_words = len(words)

            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)

            sentences = [s.strip() for s in text.split('.') if s.strip()]
            avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            unique_ratio = len(set(w.lower() for w in words)) / (n_words + 1)

            emotional_words = {
                'amazing', 'terrible', 'shocking', 'unbelievable', 'breaking',
                'urgent', 'exclusive', 'horrifying', 'incredible', 'outrageous',
                'bombshell', 'explosive', 'devastating', 'miraculous', 'insane'
            }
            emotional_ratio = sum(1 for w in words if w.lower() in emotional_words) / (n_words + 1)

            features.append([
                exclamation_count,
                question_count,
                caps_ratio,
                avg_sentence_len,
                avg_word_len,
                unique_ratio,
                emotional_ratio
            ])
        except Exception:
            features.append([0.0] * 7)

    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Clean feature engineering v2")
    parser.add_argument("--news-path", type=str, default="data/processed/news_processed.parquet")
    parser.add_argument("--graph-path", type=str, default="data/graphs_full/graph_data_clean.pt")
    parser.add_argument("--output-path", type=str, default="data/graphs_full/graph_data_clean_enhanced.pt")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip slow sentiment extraction")

    args = parser.parse_args()

    print("=" * 70)
    print("ENHANCED FEATURE ENGINEERING v2 (No Label Leakage)")
    print("=" * 70)

    # Load data
    print(f"\nLoading data...")
    news_df = pd.read_parquet(args.news_path)
    graph_data = torch.load(args.graph_path, weights_only=False)

    print(f"  Loaded {len(news_df)} articles")
    print(f"  Current features: {graph_data.x.shape}")

    # Get texts
    if 'text' in news_df.columns and news_df['text'].notna().sum() > 100:
        texts = news_df['text'].fillna('').tolist()
        print(f"  Using full text")
    elif 'title' in news_df.columns:
        texts = news_df['title'].fillna('').tolist()
        print(f"  Using titles (no full text available)")
    else:
        raise ValueError("No text or title column found")

    # Extract features
    print(f"\nExtracting enhanced features...")

    existing_np = graph_data.x.cpu().numpy()
    new_features = []

    # 1. Sentiment (slow, optional)
    if not args.skip_sentiment:
        sentiment_feats = extract_sentiment_features(texts)
        new_features.append(('sentiment', sentiment_feats))
        print(f"  Sentiment: {sentiment_feats.shape}")
    else:
        print("  Skipping sentiment (--skip-sentiment)")

    # 2. Source features (no label leakage)
    source_feats = extract_source_features(news_df)
    new_features.append(('source', source_feats))
    print(f"  Source: {source_feats.shape}")

    # 3. Named entities
    ner_feats = extract_ner_features(texts)
    new_features.append(('ner', ner_feats))
    print(f"  NER: {ner_feats.shape}")

    # 4. Readability
    readability_feats = extract_readability_features(texts)
    new_features.append(('readability', readability_feats))
    print(f"  Readability: {readability_feats.shape}")

    # 5. Writing style
    style_feats = extract_style_features(texts)
    new_features.append(('style', style_feats))
    print(f"  Style: {style_feats.shape}")

    # Combine
    print(f"\nCombining features...")
    all_new = np.hstack([f for _, f in new_features])

    # Normalize only the new features (BERT embeddings are already normalized)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_new_scaled = scaler.fit_transform(all_new)

    # Handle NaN/Inf
    all_new_scaled = np.nan_to_num(all_new_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Concatenate with original features
    combined = np.hstack([existing_np, all_new_scaled])
    enhanced_x = torch.FloatTensor(combined)

    print(f"  Original features: {graph_data.x.shape[1]}")
    print(f"  New features: {all_new_scaled.shape[1]}")
    print(f"  Total features: {enhanced_x.shape[1]}")

    # Update graph data
    graph_data.x = enhanced_x

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"\nSaving to {args.output_path}...")
    torch.save(graph_data, args.output_path)

    # Save scaler and feature names
    meta = {
        'feature_names': {name: arr.shape[1] for name, arr in new_features},
        'original_dim': existing_np.shape[1],
        'total_dim': enhanced_x.shape[1],
    }
    meta_path = args.output_path.replace('.pt', '_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n{'=' * 70}")
    print(f"DONE! Enhanced features: {enhanced_x.shape}")
    print(f"{'=' * 70}")
    for name, arr in new_features:
        print(f"  {name}: {arr.shape[1]} features")
    print(f"  Total: {enhanced_x.shape[1]} = {existing_np.shape[1]} (BERT) + {all_new_scaled.shape[1]} (new)")


if __name__ == "__main__":
    main()
