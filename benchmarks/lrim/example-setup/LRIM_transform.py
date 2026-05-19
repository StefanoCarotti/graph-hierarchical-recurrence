import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, coalesce

class ClusterData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster':
            return self.num_coarse_nodes
        if key == 'coarse_edge_index':
            return self.num_coarse_nodes  
        return super().__inc__(key, value, *args, **kwargs)

class GridBlockTransform:
    def __init__(self, block_size=2, name="GridBlockTransform"):
        self.block_size = block_size
        self.name = f"{name}_b{block_size}"

    def __call__(self, data: Data):
        data.edge_index = data.edge_index.contiguous()

        kwargs = {k: v for k, v in data}
        num_edges = data.edge_index.size(1)
        num_nodes = data.num_nodes
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            kwargs['edge_attr'] = data.edge_attr.float()
        else:
            kwargs['edge_attr'] = torch.ones((num_edges, 1), dtype=torch.float)
        
        L = int(num_nodes ** 0.5)
        if L * L != num_nodes:
            raise ValueError(f"Graph is not a perfect square grid! Nodes: {num_nodes}")
            
        if L % self.block_size != 0:
            raise ValueError(f"Grid size L={L} is not perfectly divisible by block_size={self.block_size}")

        # === GEOMETRIC COARSENING CHECK ===
        mask_0_to_1 = (data.edge_index[0] == 0) & (data.edge_index[1] == 1)
        if not mask_0_to_1.any():
            raise RuntimeError(
                "Topology validation failed: Node 0 is not connected to Node 1. "
            )

        node_indices = torch.arange(num_nodes, device=data.edge_index.device)
        x = node_indices % L
        y = node_indices // L

        coarse_x = x // self.block_size
        coarse_y = y // self.block_size
        coarse_L = L // self.block_size

        cluster = coarse_y * coarse_L + coarse_x
        kwargs['cluster'] = cluster
        
        num_coarse_nodes = coarse_L * coarse_L
        kwargs['num_coarse_nodes'] = num_coarse_nodes
        
        c_edge_index = cluster[data.edge_index]
        offset = (self.block_size - 1) / 2.0
        c_indices = torch.arange(num_coarse_nodes, device=data.edge_index.device)
        c_x = (c_indices % coarse_L).float() * self.block_size + offset
        c_y = (c_indices // coarse_L).float() * self.block_size + offset
        coarse_pos = torch.stack([c_x, c_y], dim=-1)

        c_edge_index, _ = coalesce(
            c_edge_index, 
            torch.ones(c_edge_index.size(1), 1, device=c_edge_index.device), 
            num_nodes=num_coarse_nodes
        )
        c_edge_index, _ = remove_self_loops(c_edge_index)
        
        row_c, col_c = c_edge_index        
        diff = torch.abs(coarse_pos[row_c] - coarse_pos[col_c])
        diff = torch.minimum(diff, L - diff)
        c_dist = torch.norm(diff, dim=-1, keepdim=True)

        kwargs['coarse_edge_index'] = c_edge_index.contiguous()
        kwargs['coarse_edge_attr'] = c_dist.float()
        
        return ClusterData(**kwargs)

    def __repr__(self) -> str:
        return self.name