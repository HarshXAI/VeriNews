"""Add graph statistics to the clean enhanced graph."""
import torch
import networkx as nx
import numpy as np

print("Loading enhanced graph...")
data = torch.load("data/graphs_full/graph_data_clean_enhanced.pt", map_location="cpu", weights_only=False)
print(f"Features: {data.x.shape}, Edges: {data.edge_index.shape[1]}")

# Convert to NetworkX
edge_index = data.edge_index.numpy()
G = nx.DiGraph()
G.add_nodes_from(range(data.x.shape[0]))
G.add_edges_from(zip(edge_index[0], edge_index[1]))
G_undirected = G.to_undirected()
G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
print(f"NetworkX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

features_list = []
n = data.x.shape[0]

# 1. Degree
print("1/8 Degree...")
in_deg = torch.tensor([G.in_degree(i) for i in range(n)], dtype=torch.float32).unsqueeze(1)
out_deg = torch.tensor([G.out_degree(i) for i in range(n)], dtype=torch.float32).unsqueeze(1)
features_list.extend([in_deg, out_deg, in_deg + out_deg])

# 2. Clustering
print("2/8 Clustering...")
clustering = nx.clustering(G_undirected)
features_list.append(torch.tensor([clustering[i] for i in range(n)], dtype=torch.float32).unsqueeze(1))

# 3. PageRank
print("3/8 PageRank...")
pagerank = nx.pagerank(G, max_iter=100)
features_list.append(torch.tensor([pagerank[i] for i in range(n)], dtype=torch.float32).unsqueeze(1))

# 4. Core number
print("4/8 Core number...")
core_number = nx.core_number(G_undirected)
features_list.append(torch.tensor([core_number[i] for i in range(n)], dtype=torch.float32).unsqueeze(1))

# 5. Triangles
print("5/8 Triangles...")
triangles = nx.triangles(G_undirected)
features_list.append(torch.tensor([triangles[i] for i in range(n)], dtype=torch.float32).unsqueeze(1))

# 6. Local density
print("6/8 Local density...")
local_density = []
for i in range(n):
    neighbors = list(G_undirected.neighbors(i))
    if len(neighbors) > 0:
        local_density.append(np.mean([clustering[n_] for n_ in neighbors]))
    else:
        local_density.append(0.0)
features_list.append(torch.tensor(local_density, dtype=torch.float32).unsqueeze(1))

# 7. Betweenness (sampled)
print("7/8 Betweenness centrality...")
betweenness = nx.betweenness_centrality(G, k=min(500, n))
features_list.append(torch.tensor([betweenness.get(i, 0.0) for i in range(n)], dtype=torch.float32).unsqueeze(1))

# 8. Degree centrality
print("8/8 Degree centrality...")
dc = nx.degree_centrality(G)
features_list.append(torch.tensor([dc.get(i, 0.0) for i in range(n)], dtype=torch.float32).unsqueeze(1))

# Normalize and combine
graph_stats = torch.cat(features_list, dim=1)
graph_stats = (graph_stats - graph_stats.mean(dim=0)) / (graph_stats.std(dim=0) + 1e-8)
graph_stats = torch.nan_to_num(graph_stats, nan=0.0)

enhanced = torch.cat([data.x, graph_stats], dim=1)
data.x = enhanced

torch.save(data, "data/graphs_full/graph_data_clean_full.pt")
print(f"Saved: {enhanced.shape} (403 + {graph_stats.shape[1]} = {enhanced.shape[1]})")
