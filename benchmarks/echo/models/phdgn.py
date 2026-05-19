import torch
from models.phdgn_utils import PortHamiltonianConv
import torch
from typing import Optional
from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from collections import OrderedDict


class PHDGN(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        epsilon,
        edge_dim: int = 0,
        activ_fun="tanh",
        p_conv_mode: str = "naive", # naive, gcn
        q_conv_mode: str = "naive", # naive, gcn
        doubled_dim: bool = True,
        final_state: str = "pq",
        alpha: float = 0.0,
        beta: float = 0.0,
        dampening_mode: Optional[str] = None, # param, param+, MLP4ReLU, DGNReLU
        external_mode: Optional[str] = None, # MLP4Sin, DGNtanh
        dtype=torch.float32,
        node_level_task=False,
        train_weights: bool = True,
        weight_sharing: bool = True,
        bias: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.activ_fun = activ_fun
        self.p_conv_mode = p_conv_mode
        self.q_conv_mode = q_conv_mode
        self.doubled_dim = doubled_dim
        self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.dampening_mode = dampening_mode
        self.external_mode = external_mode
        self.dtype = dtype
        self.node_level_task = node_level_task
        self.train_weights = train_weights
        self.weight_sharing = weight_sharing
        self.bias = bias

        self.emb = Linear(self.input_dim, self.hidden_dim)
        self.nhid = self.hidden_dim * 2 if self.doubled_dim else self.hidden_dim

        self.convs = ModuleList()
        for _ in range(1 if self.weight_sharing else self.num_layers):
            self.convs.append(
                PortHamiltonianConv(
                    in_channels=self.nhid,
                    edge_channels=self.edge_dim,
                    num_iters=self.num_layers if self.weight_sharing else 1,
                    epsilon=epsilon,
                    activ_fun=activ_fun,
                    p_conv_mode=p_conv_mode,
                    q_conv_mode=q_conv_mode,
                    bias=bias,
                    beta=beta,
                    alpha=alpha,
                    dampening_mode=dampening_mode,
                    external_mode=external_mode,
                    dtype=dtype,
                )
            )

        if self.final_state != "pq":
            self.nhid = self.nhid // 2

        if not train_weights:
            # for param in self.enc.parameters():
            #    param.requires_grad = False
            for param in self.convs.parameters():
                param.requires_grad = False

        self.node_level_task = node_level_task
        if self.node_level_task:
            self.readout = Sequential(
                Linear(self.nhid, self.nhid // 2),
                LeakyReLU(),
                Linear(self.nhid // 2, self.output_dim),
                LeakyReLU()
            )
        else:
            self.readout = Sequential(
                Linear(self.nhid * 3, (self.nhid * 3) // 2),
                LeakyReLU(),
                Linear((self.nhid * 3) // 2, self.output_dim),
                LeakyReLU()
            )
       

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        h = self.emb(x)

        if self.doubled_dim:
            h = torch.cat([h, h], dim=1)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)

        if self.final_state == "p":  # taking p
            h = h[:, : self.nhid]
        elif self.final_state == "q":  # taking q
            h = h[:, self.nhid :]
        else:  # self.final_state == 'pq'
            pass  # x contains both p and q already

        if not self.node_level_task:
            h = torch.cat(
                [
                    global_add_pool(h, batch),
                    global_max_pool(h, batch),
                    global_mean_pool(h, batch),
                ],
                dim=1,
            )

        h = self.readout(h)
        return h
