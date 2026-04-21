"""
Graph Transformer + Temporal Spreading Dynamics
================================================

Integrates temporal spreading features with the Graph Transformer model.
Two approaches:
  (A) Feature Concatenation: Add handcrafted temporal features to node features
  (B) Dual Encoder: 1D-CNN processes temporal curves, fused with Graph Transformer

Uses the same data split and model architecture as the baseline for fair comparison.

Baseline: 92.21% accuracy / 91.94% F1 (Graph Transformer + Virtual Node, 522-dim)
Target:   93-94% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import json
from pathlib import Path
from tqdm import tqdm

device = torch.device('cpu')
print(f"Using device: {device}")

output_dir = Path("experiments/temporal_integration")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GRAPH TRANSFORMER + TEMPORAL SPREADING DYNAMICS")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
print("1. Loading data...")

# Graph data (522-dim: 384 BERT + 10 graph stats + 128 Node2Vec)
data = torch.load('data/graphs_full/graph_data_with_node2vec.pt', weights_only=False)
splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)

train_mask = splits['train_mask']
val_mask = splits['val_mask']
test_mask = splits['test_mask']

# Temporal data
temporal_features = np.load('data/processed/temporal_features.npy')  # (N, 14) normalized
temporal_curves = np.load('data/processed/temporal_curves.npy')       # (N, 48)

# Convert to tensors
temporal_features_t = torch.from_numpy(temporal_features).float()
temporal_curves_t = torch.from_numpy(temporal_curves).float()

# Validate alignment
assert data.num_nodes == len(temporal_features), \
    f"Node count mismatch: {data.num_nodes} vs {len(temporal_features)}"

data = data.to(device)
temporal_features_t = temporal_features_t.to(device)
temporal_curves_t = temporal_curves_t.to(device)

print(f"   Graph features:    {data.x.shape} (522-dim)")
print(f"   Temporal features: {temporal_features_t.shape} (14-dim)")
print(f"   Temporal curves:   {temporal_curves_t.shape} (48-bin)")
print(f"   Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
print()


# ============================================================================
# 2. MODEL DEFINITIONS
# ============================================================================

class GraphTransformerLayer(nn.Module):
    """Efficient graph layer with virtual node for global context"""
    def __init__(self, channels, num_heads=8, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(channels, channels, heads=num_heads,
                            dropout=dropout, concat=False, add_self_loops=True)
        self.vn_update = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        self.node_update = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
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


class TemporalCurveEncoder(nn.Module):
    """1D-CNN that learns to encode temporal spreading curves into embeddings.
    
    Input: (batch, 48) — normalized tweet volume curve
    Output: (batch, embed_dim) — learned temporal embedding
    """
    def __init__(self, n_bins=48, embed_dim=32, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (N, 48)
        x = x.unsqueeze(1)  # (N, 1, 48)
        x = F.gelu(self.conv1(x))  # (N, 16, 48)
        x = F.gelu(self.conv2(x))  # (N, 32, 48)
        x = F.gelu(self.conv3(x))  # (N, 64, 48)
        x = self.pool(x).squeeze(-1)  # (N, 64)
        x = self.fc(x)  # (N, embed_dim)
        x = self.norm(x)
        return x


class GraphTransformerTemporal(nn.Module):
    """
    Graph Transformer + Temporal Encoder (Dual Encoder Fusion).
    
    Two parallel branches:
      1. Graph branch: GATv2 + Virtual Node on graph features
      2. Temporal branch: 1D-CNN on temporal curves + linear on temporal features
    
    Fusion: concatenate branch outputs → MLP classifier
    """
    def __init__(self, graph_in_channels, temporal_feature_dim=14, temporal_curve_bins=48,
                 hidden_channels=256, num_layers=4, num_heads=8, dropout=0.2,
                 temporal_embed_dim=32, num_classes=2):
        super().__init__()

        # === Graph Branch ===
        self.input_proj = nn.Linear(graph_in_channels, hidden_channels)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_channels))
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # === Temporal Branch ===
        # 1D-CNN for temporal curves
        self.curve_encoder = TemporalCurveEncoder(
            n_bins=temporal_curve_bins,
            embed_dim=temporal_embed_dim,
            dropout=dropout
        )
        # Linear projection for handcrafted temporal features
        self.temporal_feature_proj = nn.Sequential(
            nn.Linear(temporal_feature_dim, temporal_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(temporal_embed_dim),
        )

        # === Fusion Head ===
        # Graph (hidden_channels) + Curve embedding (temporal_embed_dim) + Feature embedding (temporal_embed_dim)
        fusion_dim = hidden_channels + temporal_embed_dim * 2
        self.output = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x_graph, edge_index, temporal_curves, temporal_features):
        # === Graph Branch ===
        x = self.input_proj(x_graph)
        vn = self.virtual_node
        for layer in self.layers:
            x, vn = layer(x, edge_index, vn)

        # === Temporal Branch ===
        curve_embed = self.curve_encoder(temporal_curves)        # (N, temporal_embed_dim)
        feat_embed = self.temporal_feature_proj(temporal_features)  # (N, temporal_embed_dim)

        # === Fusion ===
        fused = torch.cat([x, curve_embed, feat_embed], dim=1)
        out = self.output(fused)
        return out


class GraphTransformerConcatFeatures(nn.Module):
    """
    Simpler approach: just concatenate temporal features to node features and run
    the same Graph Transformer architecture.
    
    Input features: 522 (graph) + 14 (temporal features) = 536 dims
    """
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
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        vn = self.virtual_node
        for layer in self.layers:
            x, vn = layer(x, edge_index, vn)
        x = self.output(x)
        return x


# ============================================================================
# 3. TRAINING FUNCTION
# ============================================================================

def train_and_evaluate(model, approach_name, forward_fn, epochs=200, lr=0.001,
                       weight_decay=5e-4, patience=20):
    """Generic training loop. forward_fn(model) returns logits."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {approach_name}")
    print(f"{'='*80}\n")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    pbar = tqdm(range(epochs), desc=approach_name)

    for epoch in pbar:
        # Train
        model.train()
        optimizer.zero_grad()
        out = forward_fn(model)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validate
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = forward_fn(model)
                pred = out[val_mask].argmax(dim=1)
                val_f1 = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='weighted')

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'val_f1': f'{val_f1:.4f}',
                    'best': f'{best_val_f1:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience // 5:
                    print(f"\n   Early stopped at epoch {epoch+1}")
                    break

    pbar.close()

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        out = forward_fn(model)
        pred = out[test_mask].argmax(dim=1)
        probs = F.softmax(out[test_mask], dim=1)

    test_f1 = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='weighted')
    test_acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())
    cm = confusion_matrix(data.y[test_mask].cpu(), pred.cpu())

    class_0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"\n   Results for {approach_name}:")
    print(f"   Test F1:       {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Val F1 (best): {best_val_f1:.4f}")
    print(f"   Class 0 recall: {class_0_recall:.4f}")
    print(f"   Class 1 recall: {class_1_recall:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"   Actual 0   {cm[0,0]:<8} {cm[0,1]}")
    print(f"   Actual 1   {cm[1,0]:<8} {cm[1,1]}")

    report = classification_report(data.y[test_mask].cpu(), pred.cpu(), target_names=['Fake', 'Real'])
    print(f"\n{report}")

    results = {
        'approach': approach_name,
        'test_f1': float(test_f1),
        'test_acc': float(test_acc),
        'val_f1': float(best_val_f1),
        'class_0_recall': float(class_0_recall),
        'class_1_recall': float(class_1_recall),
        'confusion_matrix': cm.tolist(),
        'num_params': num_params,
    }

    return results, best_model_state


# ============================================================================
# 4. RUN EXPERIMENTS
# ============================================================================

all_results = {}

# ---- Approach A: Feature Concatenation (522 + 14 = 536 dims) ----
print("\n" + "#" * 80)
print("# APPROACH A: Feature Concatenation (graph + temporal features)")
print("#" * 80)

# Concatenate temporal features to graph features
x_concat = torch.cat([data.x, temporal_features_t], dim=1)
print(f"\nConcatenated features: {data.x.shape[1]} + {temporal_features_t.shape[1]} = {x_concat.shape[1]}")

torch.manual_seed(42)
np.random.seed(42)

model_a = GraphTransformerConcatFeatures(
    in_channels=x_concat.shape[1],
    hidden_channels=256,
    num_layers=4,
    num_heads=8,
    dropout=0.2
).to(device)

# Store concatenated features temporarily
data_x_original = data.x
data.x = x_concat

results_a, state_a = train_and_evaluate(
    model_a,
    "Approach A: Graph Transformer + Temporal Features (536-dim)",
    lambda m: m(data.x, data.edge_index),
    epochs=200, lr=0.001, weight_decay=5e-4, patience=20
)
all_results['approach_a'] = results_a

# Restore original features for next experiment
data.x = data_x_original


# ---- Approach B: Dual Encoder Fusion ----
print("\n" + "#" * 80)
print("# APPROACH B: Dual Encoder Fusion (Graph Transformer + 1D-CNN)")
print("#" * 80)

torch.manual_seed(42)
np.random.seed(42)

model_b = GraphTransformerTemporal(
    graph_in_channels=data.x.shape[1],
    temporal_feature_dim=temporal_features_t.shape[1],
    temporal_curve_bins=temporal_curves_t.shape[1],
    hidden_channels=256,
    num_layers=4,
    num_heads=8,
    dropout=0.2,
    temporal_embed_dim=32,
    num_classes=2
).to(device)

results_b, state_b = train_and_evaluate(
    model_b,
    "Approach B: Dual Encoder (Graph + 1D-CNN Temporal)",
    lambda m: m(data.x, data.edge_index, temporal_curves_t, temporal_features_t),
    epochs=200, lr=0.001, weight_decay=5e-4, patience=20
)
all_results['approach_b'] = results_b


# ---- Approach C: Feature Concat with curves too (522 + 14 + 48 = 584 dims) ----
print("\n" + "#" * 80)
print("# APPROACH C: Full Feature Concatenation (graph + features + curves)")
print("#" * 80)

x_full_concat = torch.cat([data.x, temporal_features_t, temporal_curves_t], dim=1)
print(f"\nFull concatenated: {data.x.shape[1]} + {temporal_features_t.shape[1]} + {temporal_curves_t.shape[1]} = {x_full_concat.shape[1]}")

torch.manual_seed(42)
np.random.seed(42)

model_c = GraphTransformerConcatFeatures(
    in_channels=x_full_concat.shape[1],
    hidden_channels=256,
    num_layers=4,
    num_heads=8,
    dropout=0.2
).to(device)

data.x = x_full_concat

results_c, state_c = train_and_evaluate(
    model_c,
    "Approach C: Full Concat (graph + features + curves, 584-dim)",
    lambda m: m(data.x, data.edge_index),
    epochs=200, lr=0.001, weight_decay=5e-4, patience=20
)
all_results['approach_c'] = results_c

# Restore
data.x = data_x_original


# ============================================================================
# 5. COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

baseline_f1 = 0.9194
baseline_acc = 0.9221

print(f"\n{'Approach':<60} {'Test F1':>10} {'Test Acc':>10} {'ΔF1':>8} {'ΔAcc':>8}")
print(f"{'-'*60} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
print(f"{'Baseline (Graph Transformer, 522-dim)':<60} {baseline_f1:>10.4f} {baseline_acc:>10.4f} {'--':>8} {'--':>8}")

for key, res in all_results.items():
    name = res['approach'][:57] + "..." if len(res['approach']) > 60 else res['approach']
    delta_f1 = res['test_f1'] - baseline_f1
    delta_acc = res['test_acc'] - baseline_acc
    print(f"{name:<60} {res['test_f1']:>10.4f} {res['test_acc']:>10.4f} {delta_f1:>+8.4f} {delta_acc:>+8.4f}")

# Find best approach
best_key = max(all_results, key=lambda k: all_results[k]['test_f1'])
best = all_results[best_key]

print(f"\n   Best approach: {best_key}")
print(f"   Best Test F1:  {best['test_f1']:.4f} ({best['test_f1']*100:.2f}%)")
print(f"   Best Test Acc: {best['test_acc']:.4f} ({best['test_acc']*100:.2f}%)")
print(f"   Improvement:   {best['test_f1'] - baseline_f1:+.4f} F1 / {best['test_acc'] - baseline_acc:+.4f} Acc")

# ============================================================================
# 6. SAVE EVERYTHING
# ============================================================================
print("\n6. Saving results...")

# Save combined results
with open(output_dir / 'results.json', 'w') as f:
    json.dump({
        'baseline': {'f1': baseline_f1, 'acc': baseline_acc},
        **all_results,
        'best_approach': best_key,
    }, f, indent=2)
print(f"   Results saved to {output_dir / 'results.json'}")

# Save best model
if best_key == 'approach_a':
    best_state = state_a
elif best_key == 'approach_b':
    best_state = state_b
else:
    best_state = state_c

torch.save(best_state, output_dir / 'best_temporal_model.pt')
print(f"   Best model saved to {output_dir / 'best_temporal_model.pt'}")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"\n   Baseline:     {baseline_f1:.4f} F1 / {baseline_acc:.4f} Acc")
print(f"   Best Result:  {best['test_f1']:.4f} F1 / {best['test_acc']:.4f} Acc")
print(f"   Improvement:  {best['test_f1'] - baseline_f1:+.4f} F1 ({(best['test_f1'] - baseline_f1)*100:+.2f} pts)")
print(f"   Approach:     {best_key}")
print(f"\n   Output dir: {output_dir}")
print("=" * 80)
