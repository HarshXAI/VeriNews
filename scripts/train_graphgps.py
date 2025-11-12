"""
GraphGPS: Graph + Global Attention + Positional Encoding + Structure-Aware
============================================================================

GraphGPS combines:
1. Local Message Passing (GATv2Conv) - captures neighborhood structure
2. Global Attention (Transformer) - captures long-range dependencies
3. Positional Encodings (Laplacian) - structural position awareness
4. Residual connections - better gradient flow

Current best: 91.49% F1 (Node2Vec ensemble)
Target: 92.5-93.5% F1 (+1.0-2.0 pts)

Reference: "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GPSConv, Linear
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp

device = torch.device('cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = Path("experiments/graphgps")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GraphGPS TRAINING")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD DATA AND COMPUTE POSITIONAL ENCODINGS
# ============================================================================
print("1. Loading data and computing positional encodings...")

data = torch.load('data/graphs_full/graph_data_with_node2vec.pt', weights_only=False)
splits = torch.load('experiments/baseline_reproduction/best_splits.pt', weights_only=False)

train_mask = splits['train_mask']
val_mask = splits['val_mask']
test_mask = splits['test_mask']

print(f"Features: {data.x.shape[1]} (394 baseline + 128 Node2Vec)")
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
print()

# Compute Laplacian Positional Encodings (LPE)
print("Computing Laplacian Positional Encodings...")

def compute_laplacian_pe(edge_index, num_nodes, k=8):
    """
    Compute Laplacian Positional Encodings using eigenvectors of graph Laplacian
    
    Args:
        edge_index: edge indices
        num_nodes: number of nodes
        k: number of eigenvectors to use
    
    Returns:
        pe: [num_nodes, k] positional encodings
    """
    # Compute normalized Laplacian
    edge_index_np = edge_index.cpu().numpy()
    edge_weight = np.ones(edge_index_np.shape[1])
    
    # Create adjacency matrix
    adj = sp.coo_matrix(
        (edge_weight, (edge_index_np[0], edge_index_np[1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Compute degree matrix
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt = sp.diags(deg_inv_sqrt)
    
    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
    laplacian = sp.eye(num_nodes) - norm_adj
    
    # Compute k smallest eigenvectors (excluding the trivial one)
    # For large graphs, use faster approximation
    try:
        from scipy.sparse.linalg import eigsh
        print(f"   Computing top {k+1} eigenvectors (this may take a few minutes)...")
        eigenvalues, eigenvectors = eigsh(
            laplacian.astype(float), 
            k=min(k+1, num_nodes-2),  # Can't compute more than n-1 eigenvectors
            which='SM',
            maxiter=1000,
            tol=1e-3  # Relaxed tolerance for faster computation
        )
        pe = eigenvectors[:, 1:k+1]  # Skip first trivial eigenvector
        if pe.shape[1] < k:
            # Pad with random features if we couldn't get enough eigenvectors
            padding = np.random.randn(num_nodes, k - pe.shape[1]) * 0.01
            pe = np.concatenate([pe, padding], axis=1)
        print(f"   ✅ Eigendecomposition complete")
    except Exception as e:
        # Fallback: use random features if eigendecomposition fails
        print(f"   Warning: Eigendecomposition failed ({e}), using random PE")
        pe = np.random.randn(num_nodes, k) * 0.1
    
    return torch.from_numpy(pe).float()

# Compute positional encodings
pe_dim = 8
pos_enc = compute_laplacian_pe(data.edge_index, data.num_nodes, k=pe_dim)
print(f"✅ Computed {pe_dim}-dim Laplacian PE")
print(f"   PE shape: {pos_enc.shape}")
print(f"   PE mean: {pos_enc.mean():.4f}, std: {pos_enc.std():.4f}")
print()

# ============================================================================
# 2. GraphGPS MODEL ARCHITECTURE
# ============================================================================

class GraphGPSLayer(nn.Module):
    """
    Single GraphGPS layer combining:
    - Local MPNN (GATv2)
    - Global Attention (Transformer)
    - Residual connections
    """
    def __init__(self, channels, num_heads=8, attn_dropout=0.0, mlp_dropout=0.0):
        super().__init__()
        
        # Local MPNN (GATv2)
        self.local_gnn = GATv2Conv(
            channels, 
            channels // num_heads,
            heads=num_heads,
            dropout=attn_dropout,
            concat=True
        )
        
        # Global Multi-Head Attention (Transformer-style)
        self.global_attn = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(mlp_dropout)
        )
        
        self.dropout = nn.Dropout(mlp_dropout)
        
    def forward(self, x, edge_index):
        # Local MPNN
        h_local = self.local_gnn(x, edge_index)
        h_local = self.dropout(h_local)
        x = self.norm1(x + h_local)  # Residual connection
        
        # Global Attention (treat all nodes as a sequence)
        h_global, _ = self.global_attn(
            x.unsqueeze(0),  # [1, N, C]
            x.unsqueeze(0),
            x.unsqueeze(0)
        )
        h_global = h_global.squeeze(0)  # [N, C]
        h_global = self.dropout(h_global)
        x = self.norm2(x + h_global)  # Residual connection
        
        # Feed-forward
        h_ffn = self.ffn(x)
        x = self.norm3(x + h_ffn)  # Residual connection
        
        return x


class GraphGPS(nn.Module):
    """
    GraphGPS: Recipe for a General, Powerful, Scalable Graph Transformer
    
    Architecture:
    1. Input projection + PE injection
    2. Multiple GraphGPS layers (local + global)
    3. Output head
    """
    def __init__(self, in_channels, pe_dim, hidden_channels=256, num_layers=4, 
                 num_heads=8, dropout=0.1, num_classes=2):
        super().__init__()
        
        # Input projection (features + positional encodings)
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.pe_encoder = Linear(pe_dim, hidden_channels)
        
        # GraphGPS layers
        self.layers = nn.ModuleList([
            GraphGPSLayer(
                hidden_channels, 
                num_heads=num_heads,
                attn_dropout=dropout,
                mlp_dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
    def forward(self, x, edge_index, pe):
        # Encode inputs
        h = self.node_encoder(x) + self.pe_encoder(pe)
        
        # Apply GraphGPS layers
        for layer in self.layers:
            h = layer(h, edge_index)
        
        # Output
        out = self.output(h)
        return out


# ============================================================================
# 3. TRAINING
# ============================================================================
print("=" * 80)
print("TRAINING GraphGPS")
print("=" * 80)
print()

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Create model
model = GraphGPS(
    in_channels=data.x.shape[1],
    pe_dim=pe_dim,
    hidden_channels=256,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
    num_classes=2
).to(device)

print(f"Model: GraphGPS (seed {seed})")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Hidden channels: 256")
print(f"Num layers: 4")
print(f"Num heads: 8")
print()

# Move data to device
data = data.to(device)
pos_enc = pos_enc.to(device)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

# Training loop
best_val_f1 = 0
best_model_state = None
patience_counter = 0
patience = 20
epochs = 300

pbar = tqdm(range(epochs), desc="Training")

for epoch in pbar:
    # Training
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, pos_enc)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, pos_enc)
            pred = out[val_mask].argmax(dim=1)
            val_f1 = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='weighted')
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'val_f1': f'{val_f1:.4f}',
                'best': f'{best_val_f1:.4f}'
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience // 10:
                pbar.set_description("Training [Early stopped]")
                break

pbar.close()

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model = model.to(device)

print(f"\n✅ Training complete!")
print(f"   Best validation F1: {best_val_f1:.4f}")

# ============================================================================
# 4. EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)
print()

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index, pos_enc)
    pred = out[test_mask].argmax(dim=1)
    probs = F.softmax(out[test_mask], dim=1)

test_f1 = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='weighted')
test_acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())

print(f"📊 Overall Performance:")
print(f"   Test F1:       {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print()

# Confusion matrix
cm = confusion_matrix(data.y[test_mask].cpu(), pred.cpu())
class_0_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

print(f"📊 Confusion Matrix:")
print(f"           Predicted")
print(f"           0      1")
print(f"Actual 0   {cm[0,0]:<6} {cm[0,1]:<6}  ({class_0_recall*100:.1f}% recall)")
print(f"       1   {cm[1,0]:<6} {cm[1,1]:<6}  ({class_1_recall*100:.1f}% recall)")
print()

print(f"📊 Per-Class Performance:")
print(f"   Class 0 recall: {class_0_recall:.4f} ({class_0_recall*100:.2f}%)")
print(f"   Class 1 recall: {class_1_recall:.4f} ({class_1_recall*100:.2f}%)")
print()

# Comparison with baselines
print("=" * 80)
print("COMPARISON WITH BASELINES")
print("=" * 80)
print()

baseline_394_ensemble = 0.9126
baseline_522_ensemble = 0.9149
baseline_class0 = 0.7581

print(f"Baseline (394 features, ensemble): {baseline_394_ensemble:.4f}")
print(f"Baseline (522 features, ensemble): {baseline_522_ensemble:.4f}")
print(f"Current (GraphGPS, 522 features):  {test_f1:.4f}")
print()
print(f"Improvement vs 522 ensemble: {test_f1 - baseline_522_ensemble:+.4f} ({(test_f1 - baseline_522_ensemble)*100:+.2f} pts)")
print()

print(f"Class 0 Recall:")
print(f"   Baseline (522 ensemble): {baseline_class0:.4f} ({baseline_class0*100:.2f}%)")
print(f"   Current (GraphGPS):      {class_0_recall:.4f} ({class_0_recall*100:.2f}%)")
print(f"   Improvement:             {class_0_recall - baseline_class0:+.4f} ({(class_0_recall - baseline_class0)*100:+.2f} pts)")
print()

# Save results
results = {
    'model': 'GraphGPS',
    'seed': seed,
    'val_f1': float(best_val_f1),
    'test_f1': float(test_f1),
    'test_acc': float(test_acc),
    'class_0_recall': float(class_0_recall),
    'class_1_recall': float(class_1_recall),
    'confusion_matrix': cm.tolist(),
    'hyperparameters': {
        'hidden_channels': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'pe_dim': pe_dim
    },
    'comparison': {
        'baseline_394_ensemble': float(baseline_394_ensemble),
        'baseline_522_ensemble': float(baseline_522_ensemble),
        'improvement': float(test_f1 - baseline_522_ensemble)
    }
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to {output_dir / 'results.json'}")
print()

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

target = 0.92
print(f"🎯 Target: {target:.4f} (92.00%)")
print(f"   Achieved: {test_f1:.4f} ({test_f1*100:.2f}%)")
print()

if test_f1 >= target:
    print("🎉 SUCCESS! Target reached!")
    gap_to_95 = 0.95 - test_f1
    print(f"   Gap to 95%: {gap_to_95:.4f} ({gap_to_95*100:.2f} pts)")
elif test_f1 >= baseline_522_ensemble:
    print("✅ Improvement over baseline!")
    gap = target - test_f1
    print(f"   Gap to 92%: {gap:.4f} ({gap*100:.2f} pts)")
else:
    print("⚠️  No improvement over baseline")

print()
print(f"📊 Key Metrics:")
print(f"   Overall F1:      {test_f1:.4f}")
print(f"   Class 0 recall:  {class_0_recall:.4f} (baseline: {baseline_class0:.4f})")
print(f"   Class 1 recall:  {class_1_recall:.4f}")
print()
print("=" * 80)
