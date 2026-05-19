import os
import random
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_cluster import radius_graph
from torch_scatter import scatter_mean, scatter_min
from torch_geometric.nn.pool import graclus
from torch_geometric.utils import add_self_loops, to_undirected


class ClusterData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster':
            return self.num_coarse_nodes
        if key == 'coarse_edge_index':
            return self.num_coarse_nodes  
        return super().__inc__(key, value, *args, **kwargs)


def get_induced_coarse_edges(fine_edge_index, cluster, edge_attr):
    """
    Constructs the coarse connectivity using scatter_min and scatter_mean 
    to calculate the aggregated features between coarse regions.
    """
    row, col = fine_edge_index
    cluster_row = cluster[row]
    cluster_col = cluster[col]
    
    mask = cluster_row != cluster_col
    coarse_edges_full = torch.stack([cluster_row[mask], cluster_col[mask]], dim=0)
    
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
            coarse_pos_list.append(torch.zeros(2))
            
    coarse_pos = torch.stack(coarse_pos_list)
    
    return final_cluster, coarse_pos, current_num_nodes


def generate_rgg_sample(num_nodes=200, radius=0.15, graclus_rounds=3, max_retries=5, self_loops=True, max_distance=None):
    for _ in range(max_retries):
        pos = torch.rand((num_nodes, 2), dtype=torch.float)
        edge_index = radius_graph(pos, r=radius, loop=False)
        
        num_edges = edge_index.size(1)
        edge_attr = torch.ones((num_edges, 1), dtype=torch.float) 
        
        G = nx.Graph() 
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.t().tolist())

        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            if len(largest_cc) < num_nodes * 0.8:
                continue 
        else:
            largest_cc = list(G.nodes())

        source_node = random.choice(list(largest_cc))
        
        try:
            dists_dict = nx.single_source_shortest_path_length(G, source_node) 
        except Exception:
            continue
            
        if max_distance is not None and max(dists_dict.values()) > max_distance:
            continue 
            
        y_distance = torch.full((num_nodes,), -1.0, dtype=torch.float)
        for node, dist in dists_dict.items():
            y_distance[node] = float(dist)
            
        mask = y_distance != -1.0
        
        is_source = torch.zeros((num_nodes, 1), dtype=torch.float)
        is_source[source_node] = 1.0
        x = is_source
        
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
    N = 50
    r = np.sqrt(13 / (2 * np.pi * N))
    data = generate_rgg_sample(num_nodes=N, radius=r, graclus_rounds=2, self_loops=False, max_distance=15)
    
    print(f"Fine Nodes: {data.num_nodes}, Fine Edges: {data.edge_index.size(1)}")
    print(f"Coarse Nodes: {data.num_coarse_nodes}, Coarse Edges: {data.coarse_edge_index.size(1)}")
    print(f"Y (Labels) Shape: {data.y.shape} (Distance to Source)")
    print(f"Max distance (hops): {data.y.max().item():.0f}")
    
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 16,
        "font.size": 14,
        "legend.fontsize": 14,
        "axes.titlesize": 18,
        "figure.titlesize": 20
    })
    
    pos = data.pos.numpy()
    edge_index = data.edge_index.numpy()
    coarse_pos = data.coarse_pos.numpy()
    coarse_edge_index = data.coarse_edge_index.numpy()
    
    distances = data.y.numpy()
    mask = data.mask.numpy()
    source_node = data.x[:, 0].nonzero(as_tuple=True)[0].item()  
    
    # Fine Graph
    G_fine = nx.Graph()
    G_fine.add_edges_from(edge_index.T.tolist())
    largest_cc_fine = max(nx.connected_components(G_fine), key=len)
    G_fine = G_fine.subgraph(largest_cc_fine) 
    fine_diameter = nx.diameter(G_fine)
    fine_eccentricity_all = nx.eccentricity(G_fine)
    avg_eccentricity = sum(fine_eccentricity_all.values()) / len(fine_eccentricity_all)
    
    # Coarse Graph 
    G_coarse = nx.Graph()
    G_coarse.add_edges_from(coarse_edge_index.T.tolist())
    G_coarse.remove_edges_from(nx.selfloop_edges(G_coarse))
    
    cluster = data.cluster.numpy()
    coarse_source_node = cluster[source_node]
    
    if len(G_coarse.nodes) > 0:
        largest_cc_coarse = max(nx.connected_components(G_coarse), key=len)
        coarse_diameter = nx.diameter(G_coarse.subgraph(largest_cc_coarse))
        coarse_eccentricity_all = nx.eccentricity(G_coarse.subgraph(largest_cc_coarse))
        avg_coarse_eccentricity = sum(coarse_eccentricity_all.values()) / len(coarse_eccentricity_all)
        
        coarse_dists_dict = nx.single_source_shortest_path_length(G_coarse, coarse_source_node)
        coarse_distances = np.zeros(data.num_coarse_nodes)
        for n, d in coarse_dists_dict.items():
            coarse_distances[n] = d
    else:
        coarse_diameter = 0
        coarse_distances = np.zeros(data.num_coarse_nodes)
        
    fine_mask = np.isfinite(distances)
    coarse_mask = np.isfinite(coarse_distances)
    
    pos = pos[fine_mask]
    edge_index = edge_index[:, fine_mask[edge_index[0]] & fine_mask[edge_index[1]]]
    distances = distances[fine_mask]
    
    coarse_pos = coarse_pos[coarse_mask]
    coarse_edge_index = coarse_edge_index[:, coarse_mask[coarse_edge_index[0]] & coarse_mask[coarse_edge_index[1]]]
    coarse_distances = coarse_distances[coarse_mask]
    
    cmap = cm.get_cmap('magma_r')  
    
    valid_dists = distances[mask]
    norm_fine = Normalize(vmin=0, vmax=valid_dists.max())
    norm_coarse = Normalize(vmin=0, vmax=coarse_distances.max())
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax1, ax2 = axes
    
    # PLOT 1: Original Fine Graph
    for u, v in edge_index.T:
        ax1.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], 
               color='#B0B0B0', linewidth=0.5, alpha=0.4, zorder=1)
    
    sc1 = ax1.scatter(pos[mask, 0], pos[mask, 1], c=distances[mask], 
                    cmap=cmap, norm=norm_fine, s=45, zorder=2, edgecolors='white', linewidths=0.3, alpha=0.9)
    
    ax1.scatter(pos[source_node, 0], pos[source_node, 1], c='#00FF00', 
               s=250, marker='*', edgecolor='black', linewidth=1, zorder=5, label="Source Node")
               
    ax1.set_title(f"Original Fine Graph\n(Nodes: {data.num_nodes}, Diameter: {fine_diameter}, Avg. Eccentricity: {avg_eccentricity:.2f})")
    ax1.legend(loc='upper right', frameon=True, shadow=False)
    ax1.axis('off')
    
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Fine Hops", rotation=270, labelpad=20)
    
    # PLOT 2: Coarse Graph Overlay
    for u, v in edge_index.T:
        ax2.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], 
               color='#D3D3D3', linewidth=0.4, alpha=0.2, zorder=1)
               
    ax2.scatter(pos[mask, 0], pos[mask, 1], c=distances[mask], 
                    cmap=cmap, norm=norm_fine, s=15, zorder=2, edgecolors='none', alpha=0.15)
    
    for u, v in coarse_edge_index.T:
        if u != v: 
            ax2.plot([coarse_pos[u, 0], coarse_pos[v, 0]], 
                    [coarse_pos[u, 1], coarse_pos[v, 1]], 
                   color='#2F4F4F', linewidth=1.8, alpha=0.9, zorder=3)
                   
    sc2 = ax2.scatter(coarse_pos[:, 0], coarse_pos[:, 1], c=coarse_distances, cmap=cmap, norm=norm_coarse, marker='o',
               s=150, edgecolor='white', linewidths=0.8, zorder=4, label="Pooled Nodes")
               
    ax2.scatter(coarse_pos[coarse_source_node, 0], coarse_pos[coarse_source_node, 1], c='#00FF00', 
               s=400, marker='*', edgecolor='black', linewidth=1, zorder=5, label="Source Cluster")
               
    ax2.set_title(f"Pooled Coarse Graph\n(Nodes: {data.num_coarse_nodes}, Diameter: {coarse_diameter}, Avg. Eccentricity: {avg_coarse_eccentricity:.2f})")
    ax2.legend(loc='upper right', frameon=True, shadow=False)
    ax2.axis('off')
    
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Coarse Hops", rotation=270, labelpad=20)
               
    plt.tight_layout()
    plt.show()

    #torch.save(data, "sample_rgg_data.pt")