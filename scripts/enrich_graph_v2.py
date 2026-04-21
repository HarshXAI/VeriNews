"""
Clean Graph Enrichment v2 - No Label Leakage
=============================================

Removes the label-based edges (fake_network, real_network) that leak
ground truth into the graph structure. Replaces them with legitimate
signals: entity co-mention, temporal proximity, and stronger content 
similarity.

Edge types:
1. Content similarity (cosine similarity of embeddings)
2. Same-source (articles from same publisher)
3. High-activity (highly shared articles)
4. Entity co-mention (articles sharing named entities)  [NEW]
5. Temporal proximity (articles in same time window)    [NEW]

NO label-based edges — the model must learn from features alone.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_content_similarity_edges(embeddings, k_similar=5, similarity_threshold=0.7, batch_size=1000):
    """Content similarity edges via cosine similarity (batched for memory)"""
    print("\n  1/5  Computing content similarity edges...")
    num_nodes = embeddings.shape[0]
    edge_list = []
    edge_weights = []

    for i in tqdm(range(0, num_nodes, batch_size), desc="  Similarity"):
        end_idx = min(i + batch_size, num_nodes)
        batch_emb = embeddings[i:end_idx]

        similarities = cosine_similarity(batch_emb, embeddings)

        for j in range(len(batch_emb)):
            node_idx = i + j
            sims = similarities[j]
            # Zero out self-similarity
            sims[node_idx] = -1.0

            # Get top k above threshold
            top_k_indices = np.argsort(sims)[::-1][:k_similar]
            for similar_idx in top_k_indices:
                sim_score = sims[similar_idx]
                if sim_score >= similarity_threshold:
                    edge_list.append([node_idx, int(similar_idx)])
                    edge_weights.append(float(sim_score))

    print(f"    Added {len(edge_list)} content similarity edges")
    return edge_list, edge_weights, ['content_similar'] * len(edge_list)


def compute_source_edges(news_df, k_source=3):
    """Same-source edges (articles from same publisher)"""
    print("\n  2/5  Adding source-based edges...")
    edge_list = []

    if 'source' not in news_df.columns:
        print("    Skipped (no source column)")
        return edge_list, [], []

    for source in tqdm(news_df['source'].unique(), desc="  Sources"):
        source_indices = news_df[news_df['source'] == source].index.tolist()

        for i, idx1 in enumerate(source_indices[:200]):
            for idx2 in source_indices[i + 1:i + k_source + 1]:
                if idx1 != idx2:
                    edge_list.append([idx1, idx2])
                    # Bidirectional
                    edge_list.append([idx2, idx1])

    print(f"    Added {len(edge_list)} same-source edges")
    return edge_list, [1.0] * len(edge_list), ['same_source'] * len(edge_list)


def compute_high_activity_edges(news_df, top_n=200, k_connect=3):
    """High-activity edges (highly shared articles)"""
    print("\n  3/5  Adding high-activity edges...")
    edge_list = []

    if 'num_tweets' not in news_df.columns:
        print("    Skipped (no num_tweets column)")
        return edge_list, [], []

    high_activity = news_df.nlargest(top_n, 'num_tweets').index.tolist()

    for i, idx1 in enumerate(high_activity):
        for idx2 in high_activity[i + 1:i + k_connect + 1]:
            if idx1 != idx2:
                edge_list.append([idx1, idx2])
                edge_list.append([idx2, idx1])

    print(f"    Added {len(edge_list)} high-activity edges")
    return edge_list, [1.0] * len(edge_list), ['high_activity'] * len(edge_list)


def compute_entity_edges(news_df, max_edges_per_entity=20):
    """
    Entity co-mention edges: articles sharing named entities.
    Uses title text to extract simple entity-like patterns.
    """
    print("\n  4/5  Adding entity co-mention edges...")
    edge_list = []

    # Extract entities from titles using simple NER-like heuristics
    # (proper NER with spaCy is slow; we use capitalized multi-word sequences)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])

        entity_to_articles = defaultdict(list)
        text_col = 'title' if 'title' in news_df.columns else 'text'
        texts = news_df[text_col].fillna('').tolist()

        for idx, text in enumerate(tqdm(texts, desc="  NER extraction")):
            if not text or len(text) < 5:
                continue
            try:
                doc = nlp(text[:500])
                for ent in doc.ents:
                    if ent.label_ in ('PERSON', 'ORG', 'GPE', 'NORP'):
                        entity_key = f"{ent.label_}:{ent.text.lower().strip()}"
                        entity_to_articles[entity_key].append(idx)
            except Exception:
                continue

        # Create edges between articles sharing entities
        for entity, articles in entity_to_articles.items():
            if 2 <= len(articles) <= max_edges_per_entity:
                for i, idx1 in enumerate(articles):
                    for idx2 in articles[i + 1:]:
                        edge_list.append([idx1, idx2])
                        edge_list.append([idx2, idx1])

        print(f"    Found {len(entity_to_articles)} unique entities")
        print(f"    Added {len(edge_list)} entity co-mention edges")

    except ImportError:
        print("    spaCy not available, using fallback capitalized-word method...")
        text_col = 'title' if 'title' in news_df.columns else 'text'
        texts = news_df[text_col].fillna('').tolist()

        word_to_articles = defaultdict(list)
        for idx, text in enumerate(texts):
            words = text.split()
            # Capitalized words > 3 chars (likely proper nouns)
            proper_nouns = set(w.strip('.,!?:;') for w in words if w[0:1].isupper() and len(w) > 3)
            for word in proper_nouns:
                word_to_articles[word.lower()].append(idx)

        for word, articles in word_to_articles.items():
            if 2 <= len(articles) <= max_edges_per_entity:
                for i, idx1 in enumerate(articles):
                    for idx2 in articles[i + 1:]:
                        edge_list.append([idx1, idx2])
                        edge_list.append([idx2, idx1])

        print(f"    Added {len(edge_list)} entity co-mention edges (fallback)")

    return edge_list, [1.0] * len(edge_list), ['entity_comention'] * len(edge_list)


def compute_temporal_edges(news_df, window_hours=72, max_per_node=5):
    """
    Temporal proximity edges: articles published close in time.
    Falls back to index-based proximity if no timestamp column.
    """
    print("\n  5/5  Adding temporal proximity edges...")
    edge_list = []

    # Check for timestamp column
    time_col = None
    for col in ['news_url', 'created_at', 'published_at', 'timestamp', 'date']:
        if col in news_df.columns:
            time_col = col
            break

    if time_col is None:
        # Fallback: use index-based proximity (articles near each other in dataset)
        print("    No timestamp column found, using index-based proximity...")
        num_nodes = len(news_df)
        window = 50  # articles within 50 indices

        for idx in tqdm(range(num_nodes), desc="  Temporal (index)"):
            start = max(0, idx - window)
            end = min(num_nodes, idx + window + 1)
            neighbors = np.random.choice(
                range(start, end), size=min(max_per_node, end - start), replace=False
            )
            for n_idx in neighbors:
                if n_idx != idx:
                    edge_list.append([idx, int(n_idx)])

    else:
        print(f"    Using timestamp column: {time_col}")
        try:
            news_df['_ts'] = pd.to_datetime(news_df[time_col], errors='coerce')
            valid_ts = news_df['_ts'].dropna()

            if len(valid_ts) < 100:
                print("    Too few valid timestamps, skipping")
                return edge_list, [], []

            sorted_indices = valid_ts.sort_values().index.tolist()
            window_td = pd.Timedelta(hours=window_hours)

            for i, idx1 in enumerate(tqdm(sorted_indices, desc="  Temporal")):
                count = 0
                for j in range(i + 1, len(sorted_indices)):
                    idx2 = sorted_indices[j]
                    if news_df.loc[idx2, '_ts'] - news_df.loc[idx1, '_ts'] > window_td:
                        break
                    if count < max_per_node:
                        edge_list.append([idx1, idx2])
                        edge_list.append([idx2, idx1])
                        count += 1

            news_df.drop('_ts', axis=1, inplace=True, errors='ignore')
        except Exception as e:
            print(f"    Timestamp parsing failed ({e}), using index-based fallback...")
            num_nodes = len(news_df)
            window = 50
            for idx in range(num_nodes):
                start = max(0, idx - window)
                end = min(num_nodes, idx + window + 1)
                neighbors = np.random.choice(
                    range(start, end), size=min(max_per_node, end - start), replace=False
                )
                for n_idx in neighbors:
                    if n_idx != idx:
                        edge_list.append([idx, int(n_idx)])

    print(f"    Added {len(edge_list)} temporal proximity edges")
    return edge_list, [1.0] * len(edge_list), ['temporal_proximity'] * len(edge_list)


def deduplicate_edges(edge_list, edge_weights, edge_types):
    """Remove duplicate edges, keeping highest weight"""
    print("\n  Deduplicating edges...")
    edge_dict = {}
    for i, (edge, weight, etype) in enumerate(zip(edge_list, edge_weights, edge_types)):
        key = (edge[0], edge[1])
        if key not in edge_dict or weight > edge_dict[key][0]:
            edge_dict[key] = (weight, etype)

    dedup_edges = []
    dedup_weights = []
    dedup_types = []
    for (src, dst), (weight, etype) in edge_dict.items():
        dedup_edges.append([src, dst])
        dedup_weights.append(weight)
        dedup_types.append(etype)

    removed = len(edge_list) - len(dedup_edges)
    print(f"    Removed {removed} duplicate edges ({len(edge_list)} -> {len(dedup_edges)})")
    return dedup_edges, dedup_weights, dedup_types


def main():
    parser = argparse.ArgumentParser(description="Clean graph enrichment v2 (no label leakage)")
    parser.add_argument("--input-graph", type=str, default="data/graphs_full/graph_data.pt")
    parser.add_argument("--output-graph", type=str, default="data/graphs_full/graph_data_clean.pt")
    parser.add_argument("--news-path", type=str, default="data/processed/news_processed.parquet")
    parser.add_argument("--k-similar", type=int, default=5)
    parser.add_argument("--k-source", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.65,
                        help="Cosine similarity threshold (lower than v1 to compensate for removed label edges)")
    parser.add_argument("--entity-max-edges", type=int, default=20)
    parser.add_argument("--temporal-window", type=int, default=72, help="Temporal window in hours")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("CLEAN GRAPH ENRICHMENT v2 (NO LABEL LEAKAGE)")
    print("=" * 70)

    # Load base graph
    print("\n Loading base graph...")
    data = torch.load(args.input_graph, weights_only=False)
    news_df = pd.read_parquet(args.news_path)

    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Original edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape[1]}")

    embeddings = data.x.numpy()

    # Build all edge types
    all_edges = []
    all_weights = []
    all_types = []

    # 1. Content similarity
    edges, weights, types = compute_content_similarity_edges(
        embeddings, k_similar=args.k_similar, similarity_threshold=args.similarity_threshold
    )
    all_edges.extend(edges)
    all_weights.extend(weights)
    all_types.extend(types)

    # 2. Same-source
    edges, weights, types = compute_source_edges(news_df, k_source=args.k_source)
    all_edges.extend(edges)
    all_weights.extend(weights)
    all_types.extend(types)

    # 3. High-activity
    edges, weights, types = compute_high_activity_edges(news_df)
    all_edges.extend(edges)
    all_weights.extend(weights)
    all_types.extend(types)

    # 4. Entity co-mention (NEW - replaces label-based edges)
    edges, weights, types = compute_entity_edges(news_df, max_edges_per_entity=args.entity_max_edges)
    all_edges.extend(edges)
    all_weights.extend(weights)
    all_types.extend(types)

    # 5. Temporal proximity (NEW - replaces label-based edges)
    edges, weights, types = compute_temporal_edges(news_df, window_hours=args.temporal_window)
    all_edges.extend(edges)
    all_weights.extend(weights)
    all_types.extend(types)

    # Deduplicate
    all_edges, all_weights, all_types = deduplicate_edges(all_edges, all_weights, all_types)

    # Convert to tensor
    if all_edges:
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_weight = torch.tensor(all_weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)

    # Update graph data
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    # Statistics
    edge_type_counts = Counter(all_types)
    print("\n" + "=" * 70)
    print("CLEAN GRAPH STATISTICS")
    print("=" * 70)
    print(f"  Total nodes: {data.x.shape[0]}")
    print(f"  Total edges: {edge_index.shape[1]}")
    print(f"  Avg degree: {edge_index.shape[1] / data.x.shape[0]:.2f}")
    print(f"\n  Edge type breakdown:")
    for edge_type, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_types) * 100
        print(f"    {edge_type}: {count:,} ({pct:.1f}%)")
    print(f"\n  NO label-based edges (fake_network, real_network) — clean graph!")

    # Save
    print(f"\n Saving clean graph...")
    os.makedirs(os.path.dirname(args.output_graph), exist_ok=True)
    torch.save(data, args.output_graph)
    print(f"  Saved to: {args.output_graph}")

    # Save metadata
    metadata = {
        'num_nodes': int(data.x.shape[0]),
        'num_edges': int(edge_index.shape[1]),
        'num_features': int(data.x.shape[1]),
        'avg_degree': float(edge_index.shape[1] / data.x.shape[0]),
        'edge_types': {k: int(v) for k, v in edge_type_counts.items()},
        'has_edge_weights': True,
        'label_leakage': False,
        'config': vars(args),
    }

    metadata_path = args.output_graph.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print("\n" + "=" * 70)
    print("CLEAN GRAPH COMPLETE!")
    print("=" * 70)
    print(f"\nNext: python scripts/train_gat_v2.py --data-path {args.output_graph}")


if __name__ == "__main__":
    main()
