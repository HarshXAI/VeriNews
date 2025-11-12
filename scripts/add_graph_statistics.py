"""
Feature Engineering - Add Graph Statistics to Node Features

Current features: 384-dim embeddings only
Enhanced features: embeddings + graph statistics

Graph statistics to add:
- Degree (in, out, total)
- Clustering coefficient
- PageRank
- Betweenness centrality
- Closeness centrality
- Core number
- Triangle count
- Local density
"""

import torch
import torch_geometric
from torch_geometric.utils import degree, to_undirected
import networkx as nx
from tqdm import tqdm
import numpy as np
from pathlib import Path

print("="*80)
print("FEATURE ENGINEERING - ADDING GRAPH STATISTICS")
print("="*80)

# Load data
data_path = "data/graphs_full/graph_data_enriched.pt"
print(f"\n📊 Loading graph data...")
data = torch.load(data_path, map_location='cpu', weights_only=False)

print(f"  Original features: {data.x.shape}")
print(f"  Nodes: {data.x.shape[0]}")
print(f"  Edges: {data.edge_index.shape[1]}")

# Convert to NetworkX for advanced statistics
print(f"\n🔄 Converting to NetworkX for feature computation...")
edge_index = data.edge_index.numpy()
G = nx.DiGraph()
G.add_nodes_from(range(data.x.shape[0]))
G.add_edges_from(zip(edge_index[0], edge_index[1]))

# Also create undirected version for some metrics
G_undirected = G.to_undirected()

# Remove self-loops for algorithms that don't support them
G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

print(f"  Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  (Self-loops removed from undirected version)")

# Compute graph statistics
print(f"\n📈 Computing graph statistics...")

features_list = []

# 1. Degree features
print("  1/8 Computing degree features...")
in_degree = torch.tensor([G.in_degree(i) for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
out_degree = torch.tensor([G.out_degree(i) for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
total_degree = in_degree + out_degree
features_list.extend([in_degree, out_degree, total_degree])
print(f"     Added 3 features (in/out/total degree)")

# 2. Clustering coefficient
print("  2/8 Computing clustering coefficient...")
clustering = nx.clustering(G_undirected)
clustering_feat = torch.tensor([clustering[i] for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
features_list.append(clustering_feat)
print(f"     Added 1 feature (clustering coefficient)")

# 3. PageRank
print("  3/8 Computing PageRank...")
pagerank = nx.pagerank(G, max_iter=100)
pagerank_feat = torch.tensor([pagerank[i] for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
features_list.append(pagerank_feat)
print(f"     Added 1 feature (PageRank)")

# 4. Core number
print("  4/8 Computing core number...")
core_number = nx.core_number(G_undirected)
core_feat = torch.tensor([core_number[i] for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
features_list.append(core_feat)
print(f"     Added 1 feature (core number)")

# 5. Triangle count
print("  5/8 Computing triangle count...")
triangles = nx.triangles(G_undirected)
triangle_feat = torch.tensor([triangles[i] for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
features_list.append(triangle_feat)
print(f"     Added 1 feature (triangle count)")

# 6. Local clustering density
print("  6/8 Computing local density...")
# Average clustering of neighbors
local_density = []
for i in tqdm(range(data.x.shape[0]), desc="     Local density", leave=False):
    neighbors = list(G_undirected.neighbors(i))
    if len(neighbors) > 0:
        avg_clustering = np.mean([clustering[n] for n in neighbors])
        local_density.append(avg_clustering)
    else:
        local_density.append(0.0)
local_density_feat = torch.tensor(local_density, dtype=torch.float32).unsqueeze(1)
features_list.append(local_density_feat)
print(f"     Added 1 feature (local density)")

# 7. Betweenness centrality (sampled for speed)
print("  7/8 Computing betweenness centrality (sampled)...")
betweenness = nx.betweenness_centrality(G, k=min(1000, data.x.shape[0]))
betweenness_feat = torch.tensor([betweenness.get(i, 0.0) for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
features_list.append(betweenness_feat)
print(f"     Added 1 feature (betweenness centrality)")

# 8. Closeness centrality (sampled for speed)
print("  8/8 Computing closeness centrality...")
# For large graphs, use approximate closeness
if G.number_of_nodes() > 5000:
    print("     Using sampled closeness for large graph...")
    sampled_nodes = np.random.choice(G.number_of_nodes(), min(1000, G.number_of_nodes()), replace=False)
    closeness = {node: 0.0 for node in range(G.number_of_nodes())}
    for node in tqdm(sampled_nodes, desc="     Closeness", leave=False):
        try:
            closeness[node] = nx.closeness_centrality(G, u=node)
        except:
            closeness[node] = 0.0
else:
    closeness = nx.closeness_centrality(G)
closeness_feat = torch.tensor([closeness.get(i, 0.0) for i in range(data.x.shape[0])], dtype=torch.float32).unsqueeze(1)
features_list.append(closeness_feat)
print(f"     Added 1 feature (closeness centrality)")

# Concatenate all features
print(f"\n🔗 Concatenating features...")
graph_stats = torch.cat(features_list, dim=1)
print(f"  Graph statistics shape: {graph_stats.shape}")

# Normalize graph statistics
print(f"\n📊 Normalizing graph statistics...")
graph_stats_normalized = (graph_stats - graph_stats.mean(dim=0)) / (graph_stats.std(dim=0) + 1e-8)

# Combine with original features
enhanced_features = torch.cat([data.x, graph_stats_normalized], dim=1)
print(f"  Original features: {data.x.shape[1]}")
print(f"  Graph statistics: {graph_stats_normalized.shape[1]}")
print(f"  Enhanced features: {enhanced_features.shape[1]}")

# Create new data object
enhanced_data = data.clone()
enhanced_data.x = enhanced_features

# Save enhanced data
output_path = "data/graphs_full/graph_data_enriched_with_stats.pt"
torch.save(enhanced_data, output_path)

print(f"\n💾 Saved enhanced graph to {output_path}")

# Summary
print(f"\n{'='*80}")
print("FEATURE ENGINEERING COMPLETE")
print(f"{'='*80}")
print(f"\nAdded {graph_stats_normalized.shape[1]} graph statistics features:")
print(f"  1. In-degree")
print(f"  2. Out-degree")
print(f"  3. Total degree")
print(f"  4. Clustering coefficient")
print(f"  5. PageRank")
print(f"  6. Core number")
print(f"  7. Triangle count")
print(f"  8. Local density")
print(f"  9. Betweenness centrality")
print(f"  10. Closeness centrality")
print(f"\nTotal features: {data.x.shape[1]} → {enhanced_features.shape[1]} (+{graph_stats_normalized.shape[1]})")
print(f"\n✅ Ready for training with enhanced features!")
