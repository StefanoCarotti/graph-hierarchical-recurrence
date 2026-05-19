import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_scatter import scatter_max
import torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        return hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        inter = int(hidden_dim * 4 / 3) # 4/3 I've seen is standard for SwiGLU
        inter = ((inter + 7) // 8) * 8
        
        self.gate_up_proj = nn.Linear(hidden_dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_dim, bias=False)
    
    @property #need these because GINEconv asks for them
    def in_features(self):
        return self.hidden_dim
    
    @property
    def out_features(self):
        return self.hidden_dim

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
class GHRModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, num_classes=10, L_steps=4, H_steps=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L_steps = L_steps 
        self.H_steps = H_steps 
        
        # State initializations
        self.H_init = nn.Parameter(torch.randn(hidden_dim)) 
        self.L_init = nn.Parameter(torch.randn(hidden_dim)) 
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)
        
        # --- Encoders ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Distance Encoders (Mapping scalar distance to vector)
        self.edge_encoder_fine = nn.Linear(edge_dim, hidden_dim) 
        self.edge_encoder_coarse = nn.Linear(1, hidden_dim)
        
        # Normalization
        self.norm_coarse = RMSNorm()
        self.norm_fine = RMSNorm()
        
        # --- H-level (Coarse) ---
        self.gnn_coarse = GINEConv(nn.Sequential(
            SwiGLU(hidden_dim), RMSNorm()), train_eps=True, edge_dim=hidden_dim)

        # --- L-level (Fine) ---
        self.gnn_fine = GINEConv(nn.Sequential(
            SwiGLU(hidden_dim), RMSNorm()), train_eps=True, edge_dim=hidden_dim)
        
        # Cross-Level Communication
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) 

        # --- Decoder
        self.classifier = nn.Linear(hidden_dim, 1) 

    
    def forward_segment(self, x_emb, edge_index, edge_attr, cluster, coarse_edge_index, coarse_pos, state):
        h_L, h_H = state
        
        # 1. Pre-calculate Edge Attributes (Distances)
        # Coarse Graph Attributes
        row_c, col_c = coarse_edge_index
        dist_c = (coarse_pos[row_c] - coarse_pos[col_c]).norm(dim=-1, keepdim=True)
        edge_attr_c = self.edge_encoder_coarse(dist_c)
        
        # Fine Graph Attributes
        edge_attr_f = self.edge_encoder_fine(edge_attr)
        
        # 2. REASONING LOOP
        for _ in range(self.H_steps):
            
            # A. Bottom-Up Aggregation (Fine -> Coarse)
            # Keeping your MPS fix: Move to CPU for scatter_max if needed
            h_L_cpu = h_L.cpu() 
            cluster_cpu = cluster.cpu()
            
            # Max pooling features from fine nodes to their cluster center
            l_summary_cpu, _ = scatter_max(h_L_cpu, cluster_cpu, dim=0, dim_size=h_H.size(0))
            l_summary = l_summary_cpu.to(h_L.device)
            
            # B. Coarse Reasoning (H-Step)
            # The Coarse GNN integrates the summary from below + its own neighbors
            msg_H = self.gnn_coarse(h_H + l_summary, coarse_edge_index, edge_attr_c)
            h_H = self.norm_coarse(h_H + msg_H)            
            
            # C. Top-Down Guidance (Coarse -> Fine)
            # Broadcast coarse state back to all fine nodes in that cluster
            guidance = h_H[cluster]
            
            # D. Fine Reasoning (L-Steps)
            for _ in range(self.L_steps):
                # Input = Previous State + Original Input + Guidance
                low_input = h_L + x_emb + self.context_proj(guidance)
                
                msg_L = self.gnn_fine(low_input, edge_index, edge_attr_f)
                h_L = self.norm_fine(h_L + msg_L)
            
        return (h_L, h_H)

    def forward(self, x, edge_index, edge_attr, cluster, coarse_edge_index, coarse_pos, state=None):
        """
        x: [N, 37] - Node features
        edge_index: [2, E]
        edge_attr: [E, edge_dim] - Edge features
        cluster: [N] - H3 Cluster IDs
        """
        x_emb = self.input_proj(x)
        
        # 1. Initialize State if First Step
        if state is None:
            # L-State: [N, H]
            h_L = self.L_init.unsqueeze(0).expand(x.size(0), -1).clone()  
            
            # H-State: [Num_Clusters, H]
            num_coarse = cluster.max().item() + 1
            h_H = self.H_init.unsqueeze(0).expand(num_coarse, -1).clone() 
            
            state = (h_L, h_H)

        # 2. Run Reasoning
        new_state = self.forward_segment(
            x_emb, edge_index, edge_attr, cluster, 
            coarse_edge_index, coarse_pos, state
        )
        
        # 3. Predict (Node Classification)
        h_L_out, _ = new_state
        
        # Direct projection from Fine State -> 10 Classes
        out = self.classifier(h_L_out)
        
        return out.squeeze(-1), new_state