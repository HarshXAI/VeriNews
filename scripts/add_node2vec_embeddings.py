"""
Add Node2Vec Embeddings to Help with Hub Nodes
==============================================

Analysis showed misclassified nodes have significantly higher:
- Core number: +0.47
- Triangle count: +0.45
- Total degree: +0.40

Node2Vec learns structural roles and should help identify hub nodes!

Expected gain: +0.4-0.8 percentage points
Target: 92.0-92.5% F1
"""

import torch
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from pathlib import Path
import time

print("=" * 80)
print("ADDING NODE2VEC EMBEDDINGS FOR HUB NODE DETECTION")
print("=" * 80)

# ============================================================================
# 1. LOAD GRAPH DATA
# ============================================================================
print("\n1. Loading graph data...")
data = torch.load('data/graphs_full/graph_data_enriched_with_stats.pt', weights_only=False)

print(f"Current features: {data.x.shape[1]} (384 original + 10 graph stats)")
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")

# ============================================================================
# 2. CONVERT TO NETWORKX
# ============================================================================
print("\n2. Converting to NetworkX for Node2Vec...")

# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(range(data.num_nodes))

# Add edges
edge_list = data.edge_index.t().cpu().numpy()
G.add_edges_from(edge_list)

print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ============================================================================
# 3. RUN NODE2VEC
# ============================================================================
print("\n3. Running Node2Vec (this may take a few minutes)...")
print("   Hyperparameters:")
print("   - Dimensions: 128")
print("   - Walk length: 80")
print("   - Context size: 10")
print("   - Walks per node: 10")
print("   - p (return): 1")
print("   - q (in-out): 1")

start_time = time.time()

# Initialize Node2Vec
node2vec = Node2Vec(
    G,
    dimensions=128,      # Embedding size
    walk_length=80,      # Length of random walks
    num_walks=10,        # Number of walks per node
    workers=4,           # Parallel workers
    p=1,                 # Return parameter
    q=1,                 # In-out parameter
    quiet=False
)

# Train embeddings
print("\nTraining Node2Vec model...")
model = node2vec.fit(
    window=10,           # Context window size
    min_count=1,         # Minimum word count
    batch_words=4        # Batch size
)

elapsed = time.time() - start_time
print(f"\n✅ Node2Vec training complete! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ============================================================================
# 4. EXTRACT EMBEDDINGS
# ============================================================================
print("\n4. Extracting embeddings...")

# Get embeddings for all nodes
embeddings = np.zeros((data.num_nodes, 128))

for node_id in range(data.num_nodes):
    try:
        embeddings[node_id] = model.wv[str(node_id)]
    except KeyError:
        # If node not in vocabulary, use zeros
        print(f"Warning: Node {node_id} not in vocabulary")
        embeddings[node_id] = np.zeros(128)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding stats:")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std:  {embeddings.std():.4f}")
print(f"  Min:  {embeddings.min():.4f}")
print(f"  Max:  {embeddings.max():.4f}")

# ============================================================================
# 5. CONCATENATE WITH EXISTING FEATURES
# ============================================================================
print("\n5. Concatenating with existing features...")

# Convert to torch tensor
node2vec_features = torch.from_numpy(embeddings).float()

# Concatenate
original_features = data.x
combined_features = torch.cat([original_features, node2vec_features], dim=1)

print(f"Original features: {original_features.shape[1]}")
print(f"Node2Vec features: {node2vec_features.shape[1]}")
print(f"Combined features: {combined_features.shape[1]} (394 + 128 = 522)")

# ============================================================================
# 6. CREATE NEW DATA OBJECT
# ============================================================================
print("\n6. Creating enhanced data object...")

from torch_geometric.data import Data

enhanced_data = Data(
    x=combined_features,
    edge_index=data.edge_index,
    y=data.y
)

# Copy any additional attributes
for key in data.keys():
    if key not in ['x', 'edge_index', 'y']:
        enhanced_data[key] = data[key]

print(f"Enhanced data created:")
print(f"  Features: {enhanced_data.x.shape}")
print(f"  Edges: {enhanced_data.edge_index.shape}")
print(f"  Labels: {enhanced_data.y.shape}")

# ============================================================================
# 7. SAVE ENHANCED DATA
# ============================================================================
print("\n7. Saving enhanced data...")

output_path = Path('data/graphs_full/graph_data_with_node2vec.pt')
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(enhanced_data, output_path)

print(f"✅ Enhanced data saved to: {output_path}")

# ============================================================================
# 8. QUICK VALIDATION
# ============================================================================
print("\n8. Validating saved data...")

# Reload and check
loaded_data = torch.load(output_path, weights_only=False)
assert loaded_data.x.shape[1] == 522, "Feature dimension mismatch!"
assert loaded_data.num_nodes == data.num_nodes, "Node count mismatch!"

print(f"✅ Validation passed!")
print(f"   - Features: {loaded_data.x.shape[1]} ✓")
print(f"   - Nodes: {loaded_data.num_nodes} ✓")
print(f"   - Edges: {loaded_data.edge_index.shape[1]} ✓")

# ============================================================================
# 9. FEATURE BREAKDOWN
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE BREAKDOWN")
print("=" * 80)

print("\nTotal features: 522")
print("\nBreakdown:")
print("  [0-384):   Original node embeddings (384 dims)")
print("  [384-394): Graph statistics (10 dims)")
print("    - in_degree, out_degree, total_degree")
print("    - clustering_coefficient, pagerank")
print("    - core_number, triangle_count")
print("    - local_density, betweenness, closeness")
print("  [394-522): Node2Vec embeddings (128 dims)")
print("    - Learned structural roles")
print("    - Should help identify hub nodes!")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("""
✅ Node2Vec embeddings added successfully!

📊 What we added:
   - 128-dimensional Node2Vec embeddings
   - Captures structural roles (hub vs peripheral)
   - Trained on full graph structure
   - Total features: 394 → 522

🎯 Expected impact:
   - Hub nodes (high degree): Better classification
   - Class 0 performance: Should improve from 73% → 80%+
   - Overall F1: 91.26% → 92.0-92.5% (+0.7-1.2 pts)

🚀 Ready to train!

Next command:
  python scripts/train_with_node2vec.py
""")

print("=" * 80)
