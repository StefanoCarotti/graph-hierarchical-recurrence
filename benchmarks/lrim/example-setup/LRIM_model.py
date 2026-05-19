import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool, global_add_pool, MessagePassing
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter
import math

def init_pure_isometry_(layer):
    """
    Standardizes a linear layer to a pure isometry (Variance = 1/fan_in).
    """
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.size(1)
        bound = math.sqrt(3.0 / fan_in)
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)
            if layer.bias is not None:
                layer.bias.zero_()
    return layer

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        inter = int(hidden_dim * 4 / 3)
        inter = ((inter + 7) // 8) * 8
        
        self.gate_up_proj = nn.Linear(hidden_dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_dim, bias=False)
        nn.init.normal_(self.gate_up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        
    @property
    def in_features(self): 
        return self.hidden_dim
        
    @property
    def out_features(self): 
        return self.hidden_dim

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def build_mlp(hidden_dim, use_swiglu=False):
    if use_swiglu:
        return SwiGLU(hidden_dim)
    else:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

class GatedGCNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout=0.1, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.activation = nn.ReLU() 
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.RMSNorm(out_dim)
        self.bn_edge_e = nn.RMSNorm(out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, e):
        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x_out, e_out = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)

        x_out = self.bn_node_x(x_out)
        e_out = self.bn_edge_e(e_out)

        x_out = self.activation(x_out)
        e_out = self.activation(e_out)

        x_out = F.dropout(x_out, self.dropout, training=self.training)
        e_out = F.dropout(e_out, self.dropout, training=self.training)

        return x_out, e_out

    def message(self, Dx_i, Ex_j, Ce):
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        self.e_ij = e_ij 
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        sum_sigma_x = sigma_ij * Bx_j
        numerator = scatter(sum_sigma_x, index, 0, None, dim_size, reduce='sum')
        denominator = scatter(sigma_ij, index, 0, None, dim_size, reduce='sum')
        return numerator / (denominator + 1e-6)

    def update(self, aggr_out, Ax):
        x_out = Ax + aggr_out
        e_out = self.e_ij
        return x_out, e_out


class GHRModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, l_steps=4, h_steps=4, use_swiglu=False, chem_task=False, conv_type='GINE', **kwargs):
        super().__init__()
        self.node_level_task = kwargs.get('node_level_task', True)
        self.conv_type = conv_type
        self.hidden_dim = hidden_dim
        self.L_steps = l_steps 
        self.H_steps = h_steps 
        
        self.H_init = nn.Parameter(torch.randn(hidden_dim))
        self.L_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        edge_in_dim = 2 if chem_task else 1
        self.edge_encoder_fine = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder_coarse = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm_coarse = RMSNorm(hidden_dim)
        self.norm_fine = RMSNorm(hidden_dim)
        
        self.step_encoder = nn.Linear(2, hidden_dim, bias=False)  

        if conv_type == 'GINE':
            mlp_coarse = build_mlp(hidden_dim, use_swiglu)       
            self.gnn_coarse = GINEConv(mlp_coarse, train_eps=True, edge_dim=hidden_dim)
            mlp_fine = build_mlp(hidden_dim, use_swiglu)
            self.gnn_fine = GINEConv(mlp_fine, train_eps=True, edge_dim=hidden_dim)
        elif conv_type == 'GatedGCN':
            self.gnn_coarse = GatedGCNLayer(hidden_dim, hidden_dim, dropout=0.0)
            self.gnn_fine = GatedGCNLayer(hidden_dim, hidden_dim, dropout=0.0)
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) 

        # Node-level regressor and graph-level regressor
        if self.node_level_task: 
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )            
        else:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward_segment(self, x_emb, data, state, edge_attr_c_init, edge_attr_f_init, step):
        h_L, h_H = state
        inc_H = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=step.device)
        inc_L = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=step.device)
        edge_attr_c = edge_attr_c_init
        edge_attr_f = edge_attr_f_init
        
        for _ in range(self.H_steps):
            h_L_norm = self.norm_fine(h_L)
            l_summary = scatter_add(h_L_norm, data.cluster, dim=0, dim_size=h_H.size(0))
            
            # --- Coarse Message Passing ---
            coarse_in = self.norm_coarse(h_H) + l_summary
            if self.conv_type == 'GINE':
                msg_H = self.gnn_coarse(coarse_in, data.coarse_edge_index, edge_attr_c)
            elif self.conv_type == 'GatedGCN':
                msg_H, edge_attr_c_new = self.gnn_coarse(coarse_in, data.coarse_edge_index, edge_attr_c)
                edge_attr_c = edge_attr_c_new + edge_attr_c_init 
            
            h_H = h_H + msg_H         
            guidance = h_H[data.cluster]
            
            step = step + inc_H
            
            for _ in range(self.L_steps):
                step_emb = self.step_encoder(step)
                low_input = self.norm_fine(h_L) + x_emb + self.context_proj(guidance) + step_emb
                
                # --- Fine Message Passing ---
                if self.conv_type == 'GINE':
                    msg_L = self.gnn_fine(low_input, data.edge_index, edge_attr_f)
                elif self.conv_type == 'GatedGCN':
                    msg_L, edge_attr_f_new = self.gnn_fine(low_input, data.edge_index, edge_attr_f)
                    edge_attr_f = edge_attr_f_new + edge_attr_f_init
                
                h_L = h_L + msg_L
                step = step + inc_L
                
        return (h_L, h_H)
                
    def forward(self, data, state=None):
        x = data.x
        x_emb = self.input_proj(x)
        edge_attr_c = self.edge_encoder_coarse(data.coarse_edge_attr)
        edge_attr_f = self.edge_encoder_fine(data.edge_attr)
        step = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=x.device)
        if state is None:
            h_L = self.L_init.unsqueeze(0).expand(x.size(0), -1).clone()  
            num_coarse = data.cluster.max().item() + 1
            h_H = self.H_init.unsqueeze(0).expand(num_coarse, -1).clone() 
            state = (h_L, h_H)
        
        new_state = self.forward_segment(x_emb, data, state, edge_attr_c, edge_attr_f, step)
        h_L_out, _ = new_state
        
        if self.node_level_task:
            return self.regressor(h_L_out).squeeze(-1), new_state
        else:
            x_max = global_max_pool(h_L_out, data.batch)
            x_mean = global_mean_pool(h_L_out, data.batch)
            x_sum = global_add_pool(h_L_out, data.batch)
            graph_emb = torch.cat([x_max, x_mean, x_sum], dim=-1)
            return self.regressor(graph_emb).squeeze(-1), new_state
        

if __name__ == "__main__":
    model = GHRModel(input_dim=10, output_dim=1, hidden_dim=32, l_steps=2, h_steps=2, node_level_task=True)
    print(model)