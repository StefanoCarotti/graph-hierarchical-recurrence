import torch
import numpy as np
import networkx as nx 
import random
from torch_geometric.data import Data, InMemoryDataset
from torch_cluster import radius_graph
from torch_scatter import scatter_mean, scatter_min
from torch_geometric.nn.pool import graclus
from torch_geometric.utils import add_self_loops, to_undirected
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class ClusterData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster':
            return self.num_coarse_nodes
        if key == 'coarse_edge_index':
            return self.num_coarse_nodes  
        return super().__inc__(key, value, *args, **kwargs)

def get_induced_coarse_edges(fine_edge_index, cluster, edge_attr):
    """
    Constructs the coarse connectivity using scatter operations to calculate the 
    total "capacity/bandwidth" between coarse regions.
    """
    row, col = fine_edge_index
    cluster_row = cluster[row]
    cluster_col = cluster[col]
    
    mask = cluster_row != cluster_col
    coarse_edges_full = torch.stack([cluster_row[mask], cluster_col[mask]], dim=0)
    
    # Handle the case where no valid coarse edges exist (e.g. graph too small/disconnected)
    if coarse_edges_full.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    coarse_edge_index, inverse_indices = torch.unique(coarse_edges_full, dim=1, return_inverse=True)
    fine_edge_attr_masked = edge_attr[mask]
    
    coarse_edge_attr_min, _ = scatter_min(fine_edge_attr_masked, inverse_indices, dim=0)
    coarse_edge_attr_mean = scatter_mean(fine_edge_attr_masked, inverse_indices, dim=0)
    coarse_edge_attr = torch.cat([coarse_edge_attr_min, coarse_edge_attr_mean], dim=1)
    return coarse_edge_index, coarse_edge_attr

def get_graclus_clusters_fixed_rounds(edge_index, pos, num_nodes, rounds=3):
    """
    Applies Graclus clustering for a fixed number of rounds.
    """
    # Graclus expects exactly undirected edges
    current_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    final_cluster = torch.arange(num_nodes, dtype=torch.long)
    current_num_nodes = num_nodes
    
    for _ in range(rounds):
        cluster = graclus(current_edge_index, num_nodes=current_num_nodes)
        
        unique_c, cluster = torch.unique(cluster, return_inverse=True)
        num_clusters = len(unique_c)
        
        if num_clusters == current_num_nodes:
            break
            
        final_cluster = cluster[final_cluster]
        
        dummy_edge_attr = torch.ones((current_edge_index.size(1), 1), dtype=torch.float)
        current_edge_index, _ = get_induced_coarse_edges(
            current_edge_index, cluster, dummy_edge_attr
        )
        current_num_nodes = num_clusters

    coarse_pos_list = []
    for i in range(current_num_nodes):
        mask = (final_cluster == i)
        if mask.any():
            coarse_pos_list.append(pos[mask].mean(dim=0))
        else:
            coarse_pos_list.append(torch.zeros(2)) # Fallback safe-guard
            
    coarse_pos = torch.stack(coarse_pos_list)
    return final_cluster, coarse_pos, current_num_nodes

def generate_rgg_sample(num_nodes=200, radius=0.15, graclus_rounds=3, max_retries=5, self_loops=True, max_distance=None):
    for retry in range(max_retries):
        pos = torch.rand((num_nodes, 2), dtype=torch.float)
        edge_index = radius_graph(pos, r=radius, loop=False)
        
        num_edges = edge_index.size(1)
        # WEIGHTED: Edges have random costs between 1.0 and 10.0
        edge_attr = torch.rand((num_edges, 1), dtype=torch.float) * 9.0 + 1.0  
        
        # DIRECTED GRAPH to handle asymmetric random weights
        G = nx.DiGraph() 
        G.add_nodes_from(range(num_nodes))
        
        edges_list = edge_index.t().tolist()
        weights_list = edge_attr.view(-1).tolist()
        
        G.add_weighted_edges_from([
            (u, v, w) for (u, v), w in zip(edges_list, weights_list)
        ])

        # Weak connectivity check
        if not nx.is_weakly_connected(G):
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            if len(largest_cc) < num_nodes * 0.8:
                continue 
        else:
            largest_cc = list(G.nodes())

        # 1. Pick Source
        source_node = random.choice(list(largest_cc))
        
        # 2. Check Hop Distance BEFORE continuing
        # networkx shortest_path_length ignores weights if weight=None, acting like BFS (hops)
        try:
            hop_dists_dict = nx.single_source_shortest_path_length(G, source_node) 
        except:
            continue
            
        if max_distance is not None and max(hop_dists_dict.values()) > max_distance:
            continue
            
        # 3. Compute Labels using Dijkstra (Weighted Distance)
        # weight='weight' forces it to use the exact continuous costs we added in add_weighted_edges_from
        try:
            dists_dict = nx.single_source_dijkstra_path_length(G, source_node, weight='weight') 
        except:
            continue
            
        y_distance = torch.full((num_nodes,), -1.0, dtype=torch.float)
        for node, dist in dists_dict.items():
            y_distance[node] = dist
            
        mask = y_distance != -1.0
        
        # 4. Features (x)
        is_source = torch.zeros((num_nodes, 1), dtype=torch.float)
        is_source[source_node] = 1.0
        x = is_source
    
        # 5. Topological clustering (Graclus)
        cluster, coarse_pos, num_coarse = get_graclus_clusters_fixed_rounds(
            edge_index, pos, num_nodes, rounds=graclus_rounds
        )
        coarse_edge_index, coarse_edge_attr = get_induced_coarse_edges(edge_index, cluster, edge_attr)
        
        if self_loops:
             edge_index, edge_attr = add_self_loops(
                edge_index, 
                edge_attr, 
                fill_value=0.0, 
                num_nodes=num_nodes
            )
             coarse_edge_index, coarse_edge_attr = add_self_loops(
                coarse_edge_index, 
                coarse_edge_attr, 
                fill_value=0.0, 
                num_nodes=num_coarse
            )
            
        return ClusterData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_distance,
            mask=mask,
            pos=pos,
            cluster=cluster,
            num_coarse_nodes=num_coarse,
            coarse_pos=coarse_pos,
            coarse_edge_index=coarse_edge_index,
            coarse_edge_attr=coarse_edge_attr,
            source_node=source_node
        )
        
    raise RuntimeError("Could not generate valid RGG")

class FixedRGGDataset(InMemoryDataset):
    def __init__(self, root=".", num_samples=1000, min_nodes=100, max_nodes=150, 
                 target_avg_degree=15, graclus_rounds=3, max_distance=None, cache_path=None):
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.target_avg_degree = target_avg_degree
        self.graclus_rounds = graclus_rounds
        self.max_distance = max_distance
        self.explicit_cache_path = cache_path
        super().__init__(root, transform=None, pre_transform=None)
        
        self.data = None
        self.slices = None

        if self.explicit_cache_path and os.path.exists(self.explicit_cache_path):
            print(f"Loading cached dataset from {self.explicit_cache_path}...")
            self.data, self.slices = torch.load(self.explicit_cache_path, weights_only=False)
            print(f"Loaded {self.len()} graphs.")
        else:
            self.process_and_save()
            
    def process_and_save(self):
        print(f"Generating {self.num_samples} graphs...")
        data_list = []
        for i in range(self.num_samples):
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} graphs...")
            while True:
                N = random.randint(self.min_nodes, self.max_nodes)
                r_base = np.sqrt(self.target_avg_degree / (2 * np.pi * N))
                r = random.uniform(r_base * 0.9, r_base * 1.1)
                
                try:
                    data = generate_rgg_sample(num_nodes=N, radius=r, 
                                              graclus_rounds=self.graclus_rounds, 
                                              self_loops=False, max_distance=self.max_distance)
                    data_list.append(data)
                    break
                except RuntimeError:
                    continue
                    
        if len(data_list) > 0:
            self.data, self.slices = self.collate(data_list)
            
            if self.explicit_cache_path:
                print(f"Saving collated dataset to {self.explicit_cache_path}...")
                torch.save((self.data, self.slices), self.explicit_cache_path)
                print("Dataset saved!")

if __name__ == "__main__":   
    # Quick test
    N = 350
    r = np.sqrt(12 / (2 * np.pi * N))
    data = generate_rgg_sample(num_nodes=N, radius=r, graclus_rounds=3, self_loops=False, max_distance=40)
    print(f"Fine Nodes: {data.num_nodes}, Fine Edges: {data.edge_index.size(1)}")
    print(f"Coarse Nodes: {data.num_coarse_nodes}, Coarse Edges: {data.coarse_edge_index.size(1)}")
    print(f"Y (Labels) Shape: {data.y.shape} (Distance to Source)")
    print(f"Max distance (Weighted): {data.y.max().item():.2f}")
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax1, ax2 = axes
    
    pos = data.pos.numpy()
    edge_index = data.edge_index.numpy()
    coarse_pos = data.coarse_pos.numpy()
    coarse_edge_index = data.coarse_edge_index.numpy()
    
    distances = data.y.numpy()
    mask = data.mask.numpy()
    source_node = data.source_node
    
    valid_dists = distances[mask]
    norm = Normalize(vmin=0, vmax=valid_dists.max())
    cmap = cm.get_cmap('viridis_r')
    
    # Original Fine Graph
    for u, v in edge_index.T:
        ax1.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], 
               color='gray', linewidth=0.5, alpha=0.3, zorder=1)
    
    sc1 = ax1.scatter(pos[mask, 0], pos[mask, 1], c=distances[mask], 
                    cmap=cmap, norm=norm, s=40, zorder=2, edgecolors='none', alpha=0.7)
    
    ax1.scatter(pos[source_node, 0], pos[source_node, 1], c='red', 
               s=200, marker='*', edgecolor='black', zorder=5, label="Source Node")
               
    ax1.set_title(f"Original Weighted Graph ({data.num_nodes} nodes)")
    ax1.axis('off')
    
    # Coarse Graph Overlay
    for u, v in edge_index.T:
        ax2.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], 
               color='gray', linewidth=0.5, alpha=0.15, zorder=1)
               
    ax2.scatter(pos[mask, 0], pos[mask, 1], c=distances[mask], 
                    cmap=cmap, norm=norm, s=20, zorder=2, edgecolors='none', alpha=0.3)
    
    for u, v in coarse_edge_index.T:
        if u != v: # Don't plot self-loops
            ax2.plot([coarse_pos[u, 0], coarse_pos[v, 0]], 
                    [coarse_pos[u, 1], coarse_pos[v, 1]], 
                   color='black', linewidth=1.5, alpha=0.8, zorder=3)
                   
    ax2.scatter(coarse_pos[:, 0], coarse_pos[:, 1], c='magenta', marker='s',
               s=120, edgecolor='black', linewidth=1.5, zorder=4, label="Coarse Nodes")
               
    ax2.scatter(pos[source_node, 0], pos[source_node, 1], c='red', 
               s=300, marker='*', edgecolor='black', zorder=5, label="Source Node")
               
    ax2.set_title(f"Coarse Graph ({data.num_coarse_nodes} nodes, Graclus x3)")
    ax2.legend(loc='lower right')
    ax2.axis('off')
               
    fig.colorbar(sc1, ax=axes.ravel().tolist(), label="Weighted Distance", shrink=0.8)
    
    plt.show()