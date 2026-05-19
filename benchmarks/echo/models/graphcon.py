import torch

from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU, ReLU, Tanh
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from typing import Optional
from collections import OrderedDict
from torch import tanh
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv


class GraphCON_GCN(Module):
    # code from https://github.com/tk-rusch/GraphCON/blob/main/src/heterophilic_graphs/models.py
    def __init__(
        self, nhid, dropout, nlayers, dt=1.0, alpha=1.0, gamma=1.0, res_version=1,
        activ_fun: str = 'tanh'
    ):
        super(GraphCON_GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.conv = GCNConv(nhid, nhid)
        self.res = Linear(nhid, nhid)
        if res_version == 1:
            self.residual = self.res_connection_v1
        else:
            self.residual = self.res_connection_v2
        self.dt = dt
        # self.act_fn = ReLU() # Original code
        self.act_fn = getattr(torch, activ_fun)  if activ_fun in dir(torch) else getattr(torch, 'tanh')
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()

    def reset_params(self):
        for name, param in self.named_parameters():
            if "weight" in name and "emb" not in name and "out" not in name:
                stdv = 1.0 / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def res_connection_v1(self, X):
        res = -self.res(self.conv.lin(X))
        return res

    def res_connection_v2(self, X):
        res = -self.conv.lin(X) + self.res(X)
        return res

    def forward(self, x, edge_index):
        input = F.dropout(x, self.dropout, training=self.training)
        # Y = self.act_fn(self.enc(input))
        Y = self.act_fn(input)
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt * (
                self.act_fn(self.conv(X, edge_index) + self.residual(X))
                - self.alpha * Y
                - self.gamma * X
            )
            X = X + self.dt * Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        return X


class GraphCON(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        iterations: int = 1,
        node_level_task: bool = False,
        epsilon: Optional[float] = None,
        activ_fun: str = 'tanh',
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = iterations
        self.dt = epsilon

        self.emb = Linear(self.input_dim, self.hidden_dim)
        
        self.conv = ModuleList()
        self.conv.append(
            GraphCON_GCN(self.hidden_dim, dropout=0.0, nlayers=self.num_layers, dt=self.dt, activ_fun=activ_fun)
        )

        self.node_level_task = node_level_task
        if self.node_level_task:
            self.readout = Sequential(
                Linear(self.hidden_dim, self.hidden_dim // 2),
                LeakyReLU(),
                Linear(self.hidden_dim // 2, self.output_dim),
                LeakyReLU()
            )
        else:
            self.readout = Sequential(
                Linear(self.hidden_dim * 3, (self.hidden_dim * 3) // 2),
                LeakyReLU(),
                Linear((self.hidden_dim * 3) // 2, self.output_dim),
                LeakyReLU()
            )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.emb(x) if self.emb else x

        for conv in self.conv:
            x = conv(x, edge_index)

        if not self.node_level_task:
            x = torch.cat(
                [
                    global_add_pool(x, batch),
                    global_max_pool(x, batch),
                    global_mean_pool(x, batch),
                ],
                dim=1,
            )
        x = self.readout(x)

        return x
