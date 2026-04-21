"""
Hypothetical Scenario: Synthetic News Inference
================================================

Generates synthetic news articles (both fake-looking and real-looking) and runs
them through our best model (Graph Transformer + Temporal, Approach A, 94.80% F1)
to demonstrate how the model classifies new articles based on their features.

Approach:
  1. Load the real graph + trained model
  2. Create synthetic articles with realistic features
  3. Inject them into the graph as new nodes with edges to existing nodes
  4. Run inference and show predictions with confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

device = torch.device('cpu')

# ============================================================================
# 1. MODEL DEFINITIONS (same as training script)
# ============================================================================

class GraphTransformerLayer(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(channels, channels, heads=num_heads,
                            dropout=dropout, concat=False, add_self_loops=True)
        self.vn_update = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(channels * 2, channels)
        )
        self.node_update = nn.Sequential(
            nn.Linear(channels * 2, channels * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(channels * 2, channels)
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.dropout = dropout

    def forward(self, x, edge_index, virtual_node):
        x_local = self.gat(x, edge_index)
        x = self.norm1(x + F.dropout(x_local, p=self.dropout, training=self.training))
        vn_input = x.mean(dim=0, keepdim=True)
        vn_update = self.vn_update(vn_input)
        virtual_node = self.norm2(virtual_node + vn_update)
        vn_broadcast = virtual_node.expand(x.size(0), -1)
        x_combined = torch.cat([x, vn_broadcast], dim=1)
        x_update = self.node_update(x_combined)
        x = self.norm3(x + x_update)
        return x, virtual_node


class GraphTransformerConcatFeatures(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=4,
                 num_heads=8, dropout=0.2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_channels))
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        vn = self.virtual_node
        for layer in self.layers:
            x, vn = layer(x, edge_index, vn)
        return self.output(x)


# ============================================================================
# 2. DEFINE SYNTHETIC SCENARIOS
# ============================================================================

# Each scenario has: title, temporal profile, expected label
SCENARIOS = [
    # --- FAKE-LOOKING ARTICLES ---
    {
        "title": "BREAKING: Scientists Confirm 5G Towers Cause COVID-19 Mutations in Humans",
        "expected": "FAKE",
        "temporal_profile": "slow_burn",
        "description": "Classic conspiracy theory with sensational language"
    },
    {
        "title": "Bill Gates Secretly Implanting Microchips Through Vaccines, Leaked Documents Reveal",
        "expected": "FAKE",
        "temporal_profile": "slow_burn",
        "description": "Anti-vaccine misinformation with conspiracy framing"
    },
    {
        "title": "EXPOSED: The Moon Landing Was Filmed in a Hollywood Studio, NASA Insider Confesses",
        "expected": "FAKE",
        "temporal_profile": "slow_burn",
        "description": "Classic conspiracy with fake insider source"
    },
    {
        "title": "Drinking Bleach Cures Cancer: The Treatment Big Pharma Doesn't Want You to Know",
        "expected": "FAKE",
        "temporal_profile": "slow_burn",
        "description": "Dangerous health misinformation"
    },
    {
        "title": "Celebrity X Found Dead in Hotel Room After Secret Satanic Ritual Gone Wrong",
        "expected": "FAKE",
        "temporal_profile": "slow_burn",
        "description": "Tabloid-style fabricated celebrity death story"
    },

    # --- REAL-LOOKING ARTICLES ---
    {
        "title": "Federal Reserve Raises Interest Rates by 0.25% Amid Inflation Concerns",
        "expected": "REAL",
        "temporal_profile": "sharp_burst",
        "description": "Standard financial news with specific factual claim"
    },
    {
        "title": "NASA's James Webb Space Telescope Captures New Images of Distant Galaxy Formation",
        "expected": "REAL",
        "temporal_profile": "sharp_burst",
        "description": "Science news from credible institution"
    },
    {
        "title": "Supreme Court Rules 6-3 in Favor of State Rights in Environmental Regulation Case",
        "expected": "REAL",
        "temporal_profile": "sharp_burst",
        "description": "Factual legal reporting with specific details"
    },
    {
        "title": "World Health Organization Reports 15% Decline in Global Malaria Deaths Since 2020",
        "expected": "REAL",
        "temporal_profile": "sharp_burst",
        "description": "Public health statistics from official source"
    },
    {
        "title": "Tesla Reports Q3 Earnings: Revenue Up 12% Year-Over-Year, Beating Analyst Estimates",
        "expected": "REAL",
        "temporal_profile": "sharp_burst",
        "description": "Corporate earnings report with specific numbers"
    },
]


def create_temporal_features(profile: str) -> np.ndarray:
    """
    Create 14 temporal features matching the fake/real propagation signatures
    we discovered in our analysis.

    Fake news pattern:  slow burn, long duration, persistent late activity
    Real news pattern:  sharp burst, quick decay, front-loaded engagement
    """
    rng = np.random.default_rng()

    if profile == "slow_burn":
        # Fake news temporal signature
        spread_duration_hours = rng.uniform(300, 700)        # Long (median ~481 hrs)
        propagation_speed     = rng.uniform(0.03, 0.15)      # Slow (median ~0.07)
        burstiness            = rng.uniform(1.5, 3.0)        # High variability
        early_ratio           = rng.uniform(0.15, 0.35)      # Less front-loaded
        late_ratio            = rng.uniform(0.18, 0.30)      # More late activity
        early_late_ratio      = early_ratio / max(late_ratio, 0.01)
        peak_bin_position     = rng.uniform(0.15, 0.45)      # Later peak
        temporal_entropy      = rng.uniform(2.5, 3.5)        # Higher entropy (spread out)
        interval_mean_hours   = rng.uniform(200, 500)        # Long gaps
        interval_std_hours    = rng.uniform(200, 600)        # Variable gaps
        interval_median_hours = rng.uniform(100, 400)
        interval_skewness     = rng.uniform(1.0, 4.0)        # Right-skewed intervals
        num_tweets_log        = rng.uniform(2.0, 4.0)        # Moderate engagement
        acceleration_first    = rng.uniform(-0.5, 0.2)       # Slow start or deceleration
    else:
        # Real news temporal signature
        spread_duration_hours = rng.uniform(20, 120)          # Short (median ~65 hrs)
        propagation_speed     = rng.uniform(0.4, 1.5)         # Fast (median ~0.76)
        burstiness            = rng.uniform(0.5, 1.5)         # Lower variability
        early_ratio           = rng.uniform(0.50, 0.80)       # Front-loaded
        late_ratio            = rng.uniform(0.03, 0.12)       # Little late activity
        early_late_ratio      = early_ratio / max(late_ratio, 0.01)
        peak_bin_position     = rng.uniform(0.0, 0.15)        # Early peak
        temporal_entropy      = rng.uniform(1.0, 2.2)         # Lower entropy (concentrated)
        interval_mean_hours   = rng.uniform(20, 80)           # Short gaps
        interval_std_hours    = rng.uniform(10, 80)           # Tighter gaps
        interval_median_hours = rng.uniform(10, 60)
        interval_skewness     = rng.uniform(0.5, 2.0)         # Less skewed
        num_tweets_log        = rng.uniform(3.0, 6.0)         # Higher engagement
        acceleration_first    = rng.uniform(0.2, 1.0)         # Strong initial burst

    return np.array([
        spread_duration_hours, propagation_speed, burstiness,
        early_ratio, late_ratio, early_late_ratio, peak_bin_position,
        temporal_entropy, interval_mean_hours, interval_std_hours,
        interval_median_hours, interval_skewness, num_tweets_log,
        acceleration_first
    ], dtype=np.float32)


# ============================================================================
# 3. MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("  HYPOTHETICAL SCENARIO: Synthetic News Classification")
    print("  Model: Graph Transformer + Temporal (Approach A, 94.80% F1)")
    print("=" * 80)
    print()

    # --- Load existing graph data ---
    print("[1/5] Loading graph data...")
    data = torch.load('data/graphs_full/graph_data_with_node2vec.pt', weights_only=False)
    temporal_features = np.load('data/processed/temporal_features.npy')
    N_original = data.x.size(0)
    D_graph = data.x.size(1)     # 522
    D_temporal = temporal_features.shape[1]  # 14
    print(f"  Graph: {N_original} nodes, {data.edge_index.size(1)} edges")
    print(f"  Features: {D_graph} graph + {D_temporal} temporal = {D_graph + D_temporal} dims")
    print()

    # --- Compute normalization stats from training data ---
    # (our temporal features need to be on the same scale as training data)
    temporal_mean = temporal_features.mean(axis=0)
    temporal_std = temporal_features.std(axis=0) + 1e-8

    # --- Generate BERT embeddings for synthetic articles ---
    print("[2/5] Generating BERT embeddings for synthetic articles...")
    model_bert = SentenceTransformer('all-MiniLM-L6-v2')
    titles = [s["title"] for s in SCENARIOS]
    bert_embeddings = model_bert.encode(titles, show_progress_bar=False)
    bert_embeddings = torch.from_numpy(bert_embeddings).float()  # (10, 384)
    print(f"  Generated {bert_embeddings.shape[0]} embeddings of dim {bert_embeddings.shape[1]}")
    print()

    # --- Create synthetic node features ---
    print("[3/5] Creating synthetic node features...")
    N_synthetic = len(SCENARIOS)

    # For graph stats (10 dims) and Node2Vec (128 dims), use average of existing nodes
    # since new articles don't have real graph structure yet
    graph_stats_mean = data.x[:, 384:394].mean(dim=0)  # avg graph stats
    node2vec_mean = data.x[:, 394:522].mean(dim=0)     # avg node2vec

    synthetic_features_list = []
    synthetic_temporal_list = []

    for i, scenario in enumerate(SCENARIOS):
        # BERT (384) + graph stats (10, averaged) + Node2Vec (128, averaged)
        graph_feat = torch.cat([
            bert_embeddings[i],
            graph_stats_mean,
            node2vec_mean,
        ])  # (522,)

        # Temporal features (14) — based on propagation profile
        raw_temporal = create_temporal_features(scenario["temporal_profile"])
        # Normalize using training data statistics
        norm_temporal = (raw_temporal - temporal_mean) / temporal_std
        norm_temporal_t = torch.from_numpy(norm_temporal).float()

        synthetic_features_list.append(graph_feat)
        synthetic_temporal_list.append(norm_temporal_t)

    synthetic_graph_features = torch.stack(synthetic_features_list)      # (10, 522)
    synthetic_temporal_features = torch.stack(synthetic_temporal_list)    # (10, 14)

    print(f"  Synthetic graph features: {synthetic_graph_features.shape}")
    print(f"  Synthetic temporal features: {synthetic_temporal_features.shape}")
    print()

    # --- Augment graph with synthetic nodes ---
    print("[4/5] Injecting synthetic nodes into graph...")
    # Combine original + synthetic features
    original_temporal_t = torch.from_numpy(temporal_features).float()
    all_graph_features = torch.cat([data.x, synthetic_graph_features], dim=0)           # (N+10, 522)
    all_temporal_features = torch.cat([original_temporal_t, synthetic_temporal_features], dim=0)  # (N+10, 14)
    all_x = torch.cat([all_graph_features, all_temporal_features], dim=1)               # (N+10, 536)

    # Create edges: connect each synthetic node to k most similar real nodes
    # using cosine similarity on BERT embeddings
    k_neighbors = 15  # connect to top-15 similar real articles
    real_bert = data.x[:, :384]  # (N, 384) BERT embeddings of real graph
    new_edges_src = []
    new_edges_dst = []

    for i in range(N_synthetic):
        syn_bert = bert_embeddings[i].unsqueeze(0)  # (1, 384)
        # Cosine similarity with all real nodes
        sims = F.cosine_similarity(syn_bert, real_bert, dim=1)  # (N,)
        topk_indices = sims.topk(k_neighbors).indices
        syn_node_idx = N_original + i
        for j in topk_indices:
            new_edges_src.extend([syn_node_idx, j.item()])
            new_edges_dst.extend([j.item(), syn_node_idx])

    new_edge_index = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)
    all_edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)  # (2, E+new)

    print(f"  Total nodes: {all_x.size(0)} ({N_original} original + {N_synthetic} synthetic)")
    print(f"  Total edges: {all_edge_index.size(1)} ({data.edge_index.size(1)} original + {new_edge_index.size(1)} new)")
    print()

    # --- Load model and run inference ---
    print("[5/5] Loading model and running inference...")
    model = GraphTransformerConcatFeatures(
        in_channels=536, hidden_channels=256, num_layers=4,
        num_heads=8, dropout=0.2, num_classes=2
    )
    state_dict = torch.load('experiments/temporal_integration/best_temporal_model.pt',
                            weights_only=False, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {num_params:,} parameters")
    print()

    with torch.no_grad():
        logits = model(all_x, all_edge_index)
        probs = F.softmax(logits, dim=1)

    # --- Display results ---
    print("=" * 80)
    print("  INFERENCE RESULTS")
    print("=" * 80)
    print()

    correct = 0
    total = N_synthetic

    for i, scenario in enumerate(SCENARIOS):
        idx = N_original + i
        fake_prob = probs[idx, 0].item()
        real_prob = probs[idx, 1].item()
        predicted = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = max(fake_prob, real_prob) * 100
        is_correct = predicted == scenario["expected"]
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"
        conf_bar = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))

        print(f"  {status} Article {i+1}: {scenario['title'][:70]}...")
        print(f"    Description:  {scenario['description']}")
        print(f"    Temporal:     {scenario['temporal_profile']}")
        print(f"    Expected:     {scenario['expected']}")
        print(f"    Predicted:    {predicted} ({confidence:.1f}% confidence)")
        print(f"    Probabilities: Fake={fake_prob:.4f}  Real={real_prob:.4f}")
        print(f"    Confidence:   [{conf_bar}] {confidence:.1f}%")
        print()

    accuracy = correct / total * 100
    print("=" * 80)
    print(f"  SUMMARY: {correct}/{total} correct ({accuracy:.0f}% accuracy on synthetic data)")
    print("=" * 80)
    print()

    # --- Also verify model on original test set ---
    print("-" * 60)
    print("  Sanity check: Original test set performance")
    print("-" * 60)
    splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)
    test_mask = splits['test_mask']
    y_true = data.y[test_mask].numpy()
    # Use only original nodes for test set evaluation
    original_logits = logits[:N_original]
    y_pred = original_logits[test_mask].argmax(dim=1).numpy()
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    print(f"  Test F1:       {f1:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  (Expected:     F1=0.9480, Acc=0.9483)")
    print()


if __name__ == "__main__":
    main()
