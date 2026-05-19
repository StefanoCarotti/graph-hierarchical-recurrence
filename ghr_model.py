import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool, global_add_pool
from torch_scatter import scatter_max, scatter_add


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

    @property  # GINEConv inspects these
    def in_features(self):
        return self.hidden_dim

    @property
    def out_features(self):
        return self.hidden_dim

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def _build_mlp(hidden_dim, use_swiglu=False):
    if use_swiglu:
        return SwiGLU(hidden_dim)
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GHRModel(nn.Module):
    """Dual-level recurrent GNN with coupled fine (h_L) and coarse (h_H) hidden states.

    Each forward call advances the joint state by h_steps outer (coarse) iterations,
    each containing l_steps inner (fine) message-passing steps.  Call T times, threading
    the returned state back in; receptive field grows with T rather than depth.

    Args:
        input_dim:       Raw node-feature dimension.  0 → OGB AtomEncoder + BondEncoder.
        output_dim:      Output dimension (node- or graph-level).
        hidden_dim:      Width of all hidden representations.
        l_steps:         Fine (L-level) message-passing steps per coarse step.
        h_steps:         Coarse (H-level) steps per forward call.
        use_swiglu:      Use SwiGLU MLP instead of Linear-SiLU-Linear.
        node_level_task:    True → per-node head; False → max+mean+sum pooling head.
        scatter_agg:        'max' or 'add' for fine→coarse aggregation.
        edge_dim_fine:      Raw fine edge feature dimension (ignored when input_dim=0).
        edge_dim_coarse:    Raw coarse edge feature dimension.
        dropout:            Dropout on message residuals and pooled embedding (0 = disabled).
        pe_dim:             Laplacian PE dimension; 0 disables the pe_proj.
        chem_task:          Shorthand: sets edge_dim_fine=2, edge_dim_coarse=2, scatter_agg='add'.
                            Also switches the node-level head to a 2-layer MLP regressor.
        linear_graph_head:  When node_level_task=False, use a flat Linear(3H, output_dim) head
                            instead of the default Sequential(3H→H, ReLU, H→1).
                            Set True for LRGB; leave False (default) for ECHO diam/energy.
                            Controls checkpoint key structure — do not change after training.
    """

    def __init__(
        self,
        input_dim: int = 0,
        output_dim: int = 1,
        hidden_dim: int = 128,
        l_steps: int = 4,
        h_steps: int = 4,
        use_swiglu: bool = False,
        node_level_task: bool = True,
        scatter_agg: str = 'max',
        edge_dim_fine: int = 1,
        edge_dim_coarse: int = 1,
        dropout: float = 0.0,
        pe_dim: int = 0,
        chem_task: bool = False,
        linear_graph_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        # backward-compat aliases (uppercase L_steps / H_steps, out_dim)
        l_steps = kwargs.pop('L_steps', l_steps)
        h_steps = kwargs.pop('H_steps', h_steps)
        output_dim = kwargs.pop('out_dim', output_dim)
        # remaining kwargs (num_layers, conv_layer, …) are intentionally ignored

        if chem_task:
            edge_dim_fine = 2
            edge_dim_coarse = 2
            scatter_agg = 'add'

        self.hidden_dim = hidden_dim
        self.L_steps = l_steps
        self.H_steps = h_steps
        self.scatter_agg = scatter_agg
        self.node_level_task = node_level_task
        self.pe_dim = pe_dim
        self.dropout = nn.Dropout(dropout)

        self.H_init = nn.Parameter(torch.randn(hidden_dim))
        self.L_init = nn.Parameter(torch.randn(hidden_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)

        # --- Input encoders ---
        if input_dim == 0:
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
                self.input_proj = AtomEncoder(emb_dim=hidden_dim)
                self.edge_encoder_fine = BondEncoder(emb_dim=hidden_dim)
            except ImportError:
                raise ImportError("input_dim=0 requires ogb: pip install ogb")
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.edge_encoder_fine = nn.Linear(edge_dim_fine, hidden_dim)

        self.edge_encoder_coarse = nn.Linear(edge_dim_coarse, hidden_dim)

        if pe_dim > 0:
            self.pe_proj = nn.Linear(pe_dim, hidden_dim)

        # --- Core GNN ---
        self.norm_coarse = RMSNorm(hidden_dim)
        self.norm_fine = RMSNorm(hidden_dim)
        self.gnn_coarse = GINEConv(_build_mlp(hidden_dim, use_swiglu), train_eps=True, edge_dim=hidden_dim)
        self.gnn_fine = GINEConv(_build_mlp(hidden_dim, use_swiglu), train_eps=True, edge_dim=hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # --- Output head ---
        # Structures below must match exactly what was used at training time (checkpoint compat).
        if node_level_task:
            if chem_task:
                # charge task: 2-layer MLP (matches original ECHO_model.py checkpoint keys)
                self.regressor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
            else:
                self.regressor = nn.Linear(hidden_dim, output_dim)
        else:
            if linear_graph_head:
                # LRGB: flat projection to output_dim (matches original LRGB_model.py keys)
                self.regressor = nn.Linear(hidden_dim * 3, output_dim)
            else:
                # ECHO diam/energy: 2-layer MLP (matches original ECHO_model.py keys)
                self.regressor = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

    def _aggregate(self, h_L_norm, cluster, num_coarse):
        if self.scatter_agg == 'add':
            return scatter_add(h_L_norm, cluster, dim=0, dim_size=num_coarse)
        out, _ = scatter_max(h_L_norm, cluster, dim=0, dim_size=num_coarse)
        return out

    def forward_segment(self, x_emb, data, state, edge_attr_c, edge_attr_f):
        """Advance (h_L, h_H) by one recurrent segment (H_steps outer × L_steps inner)."""
        h_L, h_H = state
        num_coarse = h_H.size(0)

        for _ in range(self.H_steps):
            l_summary = self._aggregate(self.norm_fine(h_L), data.cluster, num_coarse)
            msg_H = self.gnn_coarse(self.norm_coarse(h_H) + l_summary, data.coarse_edge_index, edge_attr_c)
            h_H = h_H + self.dropout(msg_H)
            guidance = h_H[data.cluster]

            for _ in range(self.L_steps):
                low_input = self.norm_fine(h_L) + x_emb + self.context_proj(guidance)
                msg_L = self.gnn_fine(low_input, data.edge_index, edge_attr_f)
                h_L = h_L + self.dropout(msg_L)

        return (h_L, h_H)

    def forward(self, data, state=None):
        x = data.x
        x_emb = self.input_proj(x)

        if self.pe_dim > 0 and hasattr(data, 'pe') and data.pe is not None:
            x_emb = x_emb + self.pe_proj(data.pe)

        edge_attr_c = self.edge_encoder_coarse(data.coarse_edge_attr)
        edge_attr_f = self.edge_encoder_fine(data.edge_attr)

        if state is None:
            h_L = self.L_init.unsqueeze(0).expand(x.size(0), -1).clone()
            num_coarse = data.cluster.max().item() + 1
            h_H = self.H_init.unsqueeze(0).expand(num_coarse, -1).clone()
            state = (h_L, h_H)

        new_state = self.forward_segment(x_emb, data, state, edge_attr_c, edge_attr_f)
        h_L_out, _ = new_state

        if self.node_level_task:
            return self.regressor(h_L_out).squeeze(-1), new_state

        batch = (
            data.batch
            if hasattr(data, 'batch') and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        x_max = global_max_pool(h_L_out, batch)
        x_mean = global_mean_pool(h_L_out, batch)
        x_sum = global_add_pool(h_L_out, batch)
        graph_emb = self.dropout(torch.cat([x_max, x_mean, x_sum], dim=-1))
        return self.regressor(graph_emb), new_state


if __name__ == "__main__":
    model = GHRModel(input_dim=10, output_dim=1, hidden_dim=32, l_steps=2, h_steps=2)
    print(model)
