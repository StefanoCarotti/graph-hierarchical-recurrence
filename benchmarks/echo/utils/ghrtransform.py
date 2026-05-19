import torch
from torch_geometric.nn import graclus, max_pool
from torch_geometric.data import Data

class ClusterData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # PyG shifts index tensors by the per-graph node count when batching;
        # cluster and coarse_edge_index must be offset by the coarse count instead.
        if key == 'cluster':
            return self.num_coarse_nodes
        if key == 'coarse_edge_index':
            return self.num_coarse_nodes
        return super().__inc__(key, value, *args, **kwargs)

class GHRTransform:
    """Pre-computes Graclus coarsening and stores the coarse graph alongside the original.

    Adds cluster (fine→coarse assignment), coarse_edge_index, coarse_edge_attr, and
    num_coarse_nodes to each Data object.  Apply once as a pre_transform or transform
    before passing data to GHRModel.
    """

    def __call__(self, data: Data):
        data.edge_index = data.edge_index.contiguous()

        kwargs = {k: v for k, v in data}
        num_edges = data.edge_index.size(1)
        
        # 1. Safely handle fine edge attributes
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # Keep the real chemical bonds!
            kwargs['edge_attr'] = data.edge_attr.float()
        else:
            # Fallback for synthetic tasks
            kwargs['edge_attr'] = torch.ones((num_edges, 1), dtype=torch.float)
        
        # 2. Graclus pure topological clustering
        cluster = graclus(data.edge_index, num_nodes=data.num_nodes)
        kwargs['cluster'] = cluster
        
        num_coarse_nodes = cluster.max().item() + 1
        kwargs['num_coarse_nodes'] = num_coarse_nodes
        
        # 3. Pool the graph AND the edge attributes simultaneously
        # By passing edge_attr into Data, max_pool will automatically coalesce/pool the coarse edges!
        pool_data = Data(edge_index=data.edge_index, edge_attr=kwargs['edge_attr'])
        pooled_data = max_pool(cluster, pool_data, transform=None)
        
        kwargs['coarse_edge_index'] = pooled_data.edge_index
        
        # 4. Safely handle coarse edge attributes
        if hasattr(pooled_data, 'edge_attr') and pooled_data.edge_attr is not None:
            kwargs['coarse_edge_attr'] = pooled_data.edge_attr.float()
        else:
            num_coarse_edges = pooled_data.edge_index.size(1)
            edge_dim = kwargs['edge_attr'].shape[1]
            kwargs['coarse_edge_attr'] = torch.ones((num_coarse_edges, edge_dim), dtype=torch.float)
        
        return ClusterData(**kwargs)