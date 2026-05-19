import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_scatter import scatter_max
from torch.nn import Parameter, init, Linear
from torch_geometric.nn import MessagePassing, GCNConv, GATv2Conv
from typing import Optional
import math
from torch_geometric.nn import GPSConv
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from ghr_model import GHRModel as _GHRModel


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6): # Pass hidden_dim here
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim)) 
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # Multiply by the learnable weight
        return (self.weight * hidden_states).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        inter = int(hidden_dim * 4 / 3) # 4/3 I've seen is standard for SwiGLU
        inter = ((inter + 7) // 8) * 8
        
        self.gate_up_proj = nn.Linear(hidden_dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_dim, bias=False)
        nn.init.normal_(self.gate_up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
    @property #need these because GINEconv asks for them
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
    

## Ablation number 1: deep vs recursive GIN 
class DeepGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=10, use_swiglu=False):
        super().__init__()
        # Init learned state like L_init
        self.state_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.state_init, std=0.02)
        
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(1, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = build_mlp(hidden_dim, use_swiglu) 
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            self.norms.append(RMSNorm(hidden_dim))
            
        self.regressor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch, state=None): 
        x_emb = self.input_proj(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        h = self.state_init.unsqueeze(0).expand(batch.x.size(0), -1).clone()
        
        for conv, norm in zip(self.convs, self.norms):
            h_in = norm(h) + x_emb
            msg = conv(h_in, batch.edge_index, edge_attr)
            h = h + msg
            
        score = self.regressor(h).squeeze(-1)
        
        # return a dummy state or None so unpacking works everywhere
        return score, None  
    

class RecursiveGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_iterations=10, use_swiglu=False):
        super().__init__()
        self.num_iterations = num_iterations
        
        self.state_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.state_init, std=0.02)
        
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(1, hidden_dim)
        
        mlp = build_mlp(hidden_dim, use_swiglu) 
        self.conv = GINEConv(mlp, train_eps=True, edge_dim=hidden_dim)
        self.norm = RMSNorm(hidden_dim)
            
        self.regressor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch, state=None): 
        x_emb = self.input_proj(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        h = self.state_init.unsqueeze(0).expand(batch.x.size(0), -1).clone()
        
        for _ in range(self.num_iterations):
            # Pre-norm + continuous injection
            h_in = self.norm(h) + x_emb
            msg = self.conv(h_in, batch.edge_index, edge_attr)
            h = h + msg
            
        return self.regressor(h).squeeze(-1), None

## Ablation number 2: reasoning loop with BPTT --> still deep vs recursive GIN

class ReasoningDeepGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=4, use_swiglu=False):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(1, hidden_dim)
        
        self.state_init = torch.nn.Parameter(torch.randn(hidden_dim))
        torch.nn.init.trunc_normal_(self.state_init, std=0.02)
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = build_mlp(hidden_dim, use_swiglu)
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            self.norms.append(RMSNorm(hidden_dim)) # Changed to RMSNorm
            
        self.regressor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch, state=None):
        # 1. Prepare raw inputs map
        x_emb = self.input_proj(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        # 2. Init State on the first reasoning step
        if state is None:
            state = self.state_init.unsqueeze(0).expand(batch.x.size(0), -1).clone()

        # 3. Track the state exactly like h_L in GHR
        h = state
        
        # 4. Pass through Deep pipeline
        for conv, norm in zip(self.convs, self.norms):
            # PRE-NORM + CONTINUOUS INJECTION 
            # (Matches exactly: low_input = self.norm_fine(h_L) + x_emb)
            h_in = norm(h) + x_emb 
            
            # GNN MESSAGE
            msg = conv(h_in, batch.edge_index, edge_attr)
            
            # LAYER RESIDUAL (h_L = h_L + msg_L)
            h = h + msg
            
        # 5. The iteratively updated 'h' IS the new state
        new_state = h
        
        # 6. Final regressor prediction from the new state
        score = self.regressor(new_state).squeeze(-1)
        
        return score, new_state


class ReasoningRecursiveGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_iterations=4, use_swiglu=False):
        super().__init__()
        self.num_iterations = num_iterations
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(1, hidden_dim)
        
        # State initialization
        self.state_init = torch.nn.Parameter(torch.randn(hidden_dim))
        torch.nn.init.trunc_normal_(self.state_init, std=0.02)
        
        mlp = build_mlp(hidden_dim, use_swiglu)
        self.conv = GINEConv(mlp, train_eps=True, edge_dim=hidden_dim)
        self.norm = RMSNorm(hidden_dim) # Changed to RMSNorm
        self.regressor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch, state=None):
        x_emb = self.input_proj(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        if state is None:
            state = self.state_init.unsqueeze(0).expand(batch.x.size(0), -1).clone()

        h = state
        
        # Apply the singular GIN layer recursively for N iterations
        for _ in range(self.num_iterations):
            # PRE-NORM + CONTINUOUS INJECTION 
            h_in = self.norm(h) + x_emb 
            
            # GNN MESSAGE
            msg = self.conv(h_in, batch.edge_index, edge_attr)
            
            # LAYER RESIDUAL
            h = h + msg
            
        # The iteratively updated 'h' IS the new state for BPTT
        new_state = h
        
        score = self.regressor(new_state).squeeze(-1)
        
        return score, new_state
    


class GHR(_GHRModel):
    """GHR for weighted RGG: coarse edges carry 2-feature vectors, fine edges carry 1."""

    def __init__(self, input_dim, hidden_dim, L_steps=4, H_steps=4, use_swiglu=False, **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            l_steps=L_steps,
            h_steps=H_steps,
            use_swiglu=use_swiglu,
            edge_dim_coarse=2,
            **kwargs,
        )



class NaiveAggr(MessagePassing):
    r"""
    Simple graph convolution which compute a transformation of
    neighboring nodes:  sum_{j \in N(u)} Vx_j
    """

    def __init__(self, 
                 in_channels, 
                 edge_channels: int = 0):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.lin = Linear(in_channels, in_channels, bias=False)
        self.edge_lin = None
        if edge_channels > 0:
            self.edge_lin = Linear(edge_channels, in_channels)
        self.reset_parameters()

    def forward(self, x, edge_index=None, edge_attr=None):
        out = self.propagate(
            x=self.lin(x), edge_index=edge_index, edge_attr=edge_attr
        )
        return out

    def message(
        self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_attr is None:
            return x_j
        elif self.edge_lin is None:
            if len(edge_attr.shape) == 1:
                return edge_attr.view(-1, 1) * x_j
            else:
                raise ValueError()
        else:
            return x_j + self.edge_lin(edge_attr)
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.edge_lin is not None: self.edge_lin.reset_parameters()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels: {self.in_channels}, edge_channels: {self.edge_channels})"

conv_names = ["NaiveAggr", "GCNConv"]
class AntiSymmetricConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        edge_channels: int = 0,
        num_iters: int = 1,
        gamma: float = 0.1,
        epsilon: float = 0.1,
        activ_fun: str = "tanh",  # it should be monotonically non-decreasing
        graph_conv: str = "NaiveAggr",
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="add")
        self.W = Parameter(torch.empty((in_channels, in_channels)))
        self.bias = Parameter(torch.empty(in_channels)) if bias else None

        if graph_conv == "NaiveAggr":
            self.conv = NaiveAggr(in_channels, edge_channels=edge_channels)
        elif graph_conv == "GCNConv":
            self.conv = GCNConv(in_channels, in_channels, bias=False)
        else:
            raise NotImplementedError(
                f"{graph_conv} not implemented. {graph_conv} is not in {conv_names}"
            )

        self.graph_conv = graph_conv
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)
        self.activ_fun = activ_fun

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.antisymmetric_W = (
            self.W
            - self.W.T
            - self.gamma * torch.eye(self.in_channels, device=self.W.device)
        )
        out = x
        for _ in range(self.num_iters):
            neigh_x = self.conv(
                out, edge_index=edge_index,
                **({'edge_attr': edge_attr} if self.graph_conv == 'NaiveAggr' else {})
            )
            conv = out @ self.antisymmetric_W.T + neigh_x
            if external_input is not None:
                conv = conv + external_input
            if self.bias is not None:
                conv += self.bias
            out = conv
        return self.epsilon * self.activation(out)

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        self.conv.reset_parameters()

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    

class GHRAntiSymmetric(nn.Module):
    def __init__(self, input_dim, hidden_dim, L_steps=4, H_steps=4, use_swiglu=False, epsilon=0.1, gamma=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L_steps = L_steps
        self.H_steps = H_steps
        
        self.H_init = nn.Parameter(torch.randn(hidden_dim)) 
        self.L_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)
        
        # Encoders
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Normalization layers (CRITICAL for the deep learning stability of this approach)
        self.norm_coarse = RMSNorm(hidden_dim)
        self.norm_fine = RMSNorm(hidden_dim)
        
        # H-level Conv (edge_channels=0 to trigger 1D scalar multiplication in NaiveAggr)
        self.gnn_coarse = AntiSymmetricConv(
            in_channels=hidden_dim, 
            edge_channels=0,
            num_iters=1, 
            epsilon=epsilon,
            gamma=gamma,
            activ_fun='tanh',
            graph_conv="NaiveAggr"
        )
        
        # L-level Conv 
        self.gnn_fine = AntiSymmetricConv(
            in_channels=hidden_dim, 
            edge_channels=0,
            num_iters=1, 
            epsilon=epsilon,
            gamma=gamma,
            activ_fun='tanh',
            graph_conv="NaiveAggr"
        )
        
        # Receives Guidance from H-level
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.regressor = nn.Linear(hidden_dim, 1, bias=True)

    def forward_segment(self, x_emb, data, state, edge_weight_c, edge_weight_f):
        h_L, h_H = state
        
        for _ in range(self.H_steps):
            # ZOOM OUT (L -> H)
            h_L_norm = self.norm_fine(h_L) 
            l_summary, _ = scatter_max(h_L_norm, data.cluster, dim=0, dim_size=h_H.size(0))
            
            # Guidance (H-Step)
            # Input bundles the normalized state and the summary
            h_H_in = self.norm_coarse(h_H) + l_summary
            msg_H = self.gnn_coarse(h_H_in, data.coarse_edge_index, edge_attr=edge_weight_c)
            h_H = h_H + msg_H 
            guidance = h_H[data.cluster]
            
            # INNER LOOP (L-Steps)
            for _ in range(self.L_steps):
                # Input bundles the normalized state, features, and context
                low_input = self.norm_fine(h_L) + x_emb + self.context_proj(guidance)
                msg_L = self.gnn_fine(low_input, data.edge_index, edge_attr=edge_weight_f)
                h_L = h_L + msg_L 
            
        return (h_L, h_H)

    def forward(self, data, state=None):
        x = data.x
        x_emb = self.input_proj(x)
        
        # 1D Edge Weights for Scalar Broadcast
        edge_weight_f = data.edge_attr.squeeze(-1) if hasattr(data, 'edge_attr') else None
        edge_weight_c = data.coarse_edge_attr[:, 1] if hasattr(data, 'coarse_edge_attr') else None
        
        if state is None:
            h_L = self.L_init.unsqueeze(0).expand(x.size(0), -1).clone()  
            num_coarse = data.cluster.max().item() + 1
            h_H = self.H_init.unsqueeze(0).expand(num_coarse, -1).clone() 
            state = (h_L, h_H)

        new_state = self.forward_segment(x_emb, data, state, edge_weight_c, edge_weight_f)
        
        h_L_out, _ = new_state        
        return self.regressor(h_L_out).squeeze(), new_state
    


class GHRGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, L_steps=4, H_steps=4, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L_steps = L_steps
        self.H_steps = H_steps
        
        self.H_init = nn.Parameter(torch.randn(hidden_dim)) 
        self.L_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.norm_coarse = RMSNorm(hidden_dim)
        self.norm_fine = RMSNorm(hidden_dim)
        
        # Standard GCN layers
        self.gnn_coarse = GCNConv(hidden_dim, hidden_dim)
        self.gnn_fine = GCNConv(hidden_dim, hidden_dim)
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.regressor = nn.Linear(hidden_dim, 1, bias=True)

    def forward_segment(self, x_emb, data, state, edge_weight_c, edge_weight_f):
        h_L, h_H = state
        
        for _ in range(self.H_steps):
            h_L_norm = self.norm_fine(h_L) 
            l_summary, _ = scatter_max(h_L_norm, data.cluster, dim=0, dim_size=h_H.size(0))
            
            # GCN Coarse
            h_H_in = self.norm_coarse(h_H) + l_summary
            msg_H = self.gnn_coarse(h_H_in, data.coarse_edge_index, edge_weight_c)
            h_H = h_H + F.relu(msg_H) # Include residual addition and activation
            
            guidance = h_H[data.cluster]
            
            for _ in range(self.L_steps):
                low_input = self.norm_fine(h_L) + x_emb + self.context_proj(guidance)
                msg_L = self.gnn_fine(low_input, data.edge_index, edge_weight_f)
                h_L = h_L + F.relu(msg_L)
            
        return (h_L, h_H)

    def forward(self, data, state=None):
        x = data.x
        x_emb = self.input_proj(x)
        
        # 1D weights for GCN
        edge_weight_f = data.edge_attr.squeeze(-1) if hasattr(data, 'edge_attr') else None
        edge_weight_c = data.coarse_edge_attr[:, 1] if hasattr(data, 'coarse_edge_attr') else None
        
        if state is None:
            h_L = self.L_init.unsqueeze(0).expand(x.size(0), -1).clone()  
            num_coarse = data.cluster.max().item() + 1
            h_H = self.H_init.unsqueeze(0).expand(num_coarse, -1).clone() 
            state = (h_L, h_H)

        new_state = self.forward_segment(x_emb, data, state, edge_weight_c, edge_weight_f)
        h_L_out, _ = new_state        
        return self.regressor(h_L_out).squeeze(), new_state
    

class GHRGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, L_steps=4, H_steps=4, heads=1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L_steps = L_steps
        self.H_steps = H_steps
        
        self.H_init = nn.Parameter(torch.randn(hidden_dim)) 
        self.L_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder_fine = nn.Linear(1, hidden_dim) 
        self.edge_encoder_coarse = nn.Linear(2, hidden_dim)
        
        self.norm_coarse = RMSNorm(hidden_dim)
        self.norm_fine = RMSNorm(hidden_dim)
        
        # GATv2 for dynamic attention, allows encoded edge_attr injection
        self.gnn_coarse = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim)
        self.gnn_fine = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim)
        
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.regressor = nn.Linear(hidden_dim, 1, bias=True)

    def forward_segment(self, x_emb, data, state, edge_attr_c, edge_attr_f):
        h_L, h_H = state
        
        for _ in range(self.H_steps):
            h_L_norm = self.norm_fine(h_L) 
            l_summary, _ = scatter_max(h_L_norm, data.cluster, dim=0, dim_size=h_H.size(0))
            
            # GAT Coarse
            h_H_in = self.norm_coarse(h_H) + l_summary
            msg_H = self.gnn_coarse(h_H_in, data.coarse_edge_index, edge_attr_c)
            h_H = h_H + F.relu(msg_H)
            
            guidance = h_H[data.cluster]
            
            for _ in range(self.L_steps):
                low_input = self.norm_fine(h_L) + x_emb + self.context_proj(guidance)
                msg_L = self.gnn_fine(low_input, data.edge_index, edge_attr_f)
                h_L = h_L + F.relu(msg_L)
            
        return (h_L, h_H)

    def forward(self, data, state=None):
        x = data.x
        x_emb = self.input_proj(x)
        
        # Project sparse edges to dense representations for GATv2
        edge_attr_c = self.edge_encoder_coarse(data.coarse_edge_attr)
        edge_attr_f = self.edge_encoder_fine(data.edge_attr)
        
        if state is None:
            h_L = self.L_init.unsqueeze(0).expand(x.size(0), -1).clone()  
            num_coarse = data.cluster.max().item() + 1
            h_H = self.H_init.unsqueeze(0).expand(num_coarse, -1).clone() 
            state = (h_L, h_H)

        new_state = self.forward_segment(x_emb, data, state, edge_attr_c, edge_attr_f)
        h_L_out, _ = new_state        
        return self.regressor(h_L_out).squeeze(), new_state
    


class SimpleGPS(torch.nn.Module):
    """
    A lightweight GraphGPS model tailored for SSSP.
    Combines local GINEConv (to process edge weights/topology) 
    with global Multi-Head Attention (to capture long-range routing).
    """
    def __init__(self, input_dim, hidden_dim=16, num_layers=8, heads=4, use_swiglu=False, dropout=0.3):
        super().__init__()
        
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Assuming fine graphs have 1D edge features
        self.edge_encoder = nn.Linear(1, hidden_dim) 
        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            # Local Message Passing (GINE to handle edge attributes)
            mlp = build_mlp(hidden_dim, use_swiglu)
            local_conv = GINEConv(mlp, train_eps=True, edge_dim=hidden_dim)
            
            # GPS wraps the local MPNN and a global Multi-Head Attention layer.
            # Using PyG's built in dense attention mechanism natively handles the dropout properly.
            self.layers.append(GPSConv(hidden_dim, conv=local_conv, heads=heads, dropout=self.dropout))
            self.norms.append(RMSNorm(hidden_dim))
            
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, batch, state=None):
        h = self.input_proj(batch.x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Flatten edge_attr if necessary, similar to other models
        edge_attr_raw = batch.edge_attr.view(-1, 1) if batch.edge_attr.dim() == 1 else batch.edge_attr
        edge_attr = self.edge_encoder(edge_attr_raw)
        
        # Process through GPS Layers
        for gps_layer, norm in zip(self.layers, self.norms):
            # Normalization before the layer (standard for transformers/GPS)
            h_in = norm(h)
            
            # GPSConv natively requires the `batch.batch` vector to isolate 
            # the global attention strictly within individual graphs in the batch.
            h = h + gps_layer(h_in, batch.edge_index, batch=batch.batch, edge_attr=edge_attr)
            
        score = self.regressor(h).squeeze(-1)
        
        # Return a dummy state for compatibility with your training loop
        return score, None
    

class SimpleDeepGIN(torch.nn.Module):
    """
    A standard Deep GIN model serving as a direct ablation baseline.
    Processes input features directly without recurrent/steady-state initialization.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=4, use_swiglu=False, dropout=0.3):
        super().__init__()
        
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Assuming fine graphs have 1D edge features
        self.edge_encoder = nn.Linear(1, hidden_dim) 
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = build_mlp(hidden_dim, use_swiglu)
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            self.norms.append(RMSNorm(hidden_dim))
            
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, batch, state=None):
        h = self.input_proj(batch.x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Flatten edge_attr if necessary, matching SimpleGPS
        edge_attr_raw = batch.edge_attr.view(-1, 1) if batch.edge_attr.dim() == 1 else batch.edge_attr
        edge_attr = self.edge_encoder(edge_attr_raw)
        
        # Standard forward pass with pre-normalization and residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_in = norm(h)
            msg = conv(h_in, batch.edge_index, edge_attr)
            
            # Apply dropout to the message before adding it to the residual stream
            msg = F.dropout(msg, p=self.dropout, training=self.training)
            h = h + msg
            
        score = self.regressor(h).squeeze(-1)
        
        # Return a dummy state for compatibility with your training loop
        return score, None