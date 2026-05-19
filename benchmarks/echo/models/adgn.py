import torch

from torch.nn import Module, Parameter, init, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.nn import (
    MessagePassing,
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.data import Data
from typing import Optional
import math
from collections import OrderedDict


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
        return f"self.__class__.__name__(in_channels: {self.in_channels}, edge_channels: {self.edge_channels})"


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
            NotImplementedError(
                f"{graph_conv} not implemented. {graph_conv} is not in {conv_names}"
            )

        self.graph_conv = graph_conv
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, 'tanh')
        self.activ_fun = activ_fun

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        self.antisymmetric_W = (
            self.W
            - self.W.T
            - self.gamma * torch.eye(self.in_channels, device=self.W.device)
        )
        for i in range(self.num_iters):
            neigh_x = self.conv(x, edge_index=edge_index, 
                                **({'edge_attr': edge_attr} if self.graph_conv == 'NaiveAggr' else {}))
            conv = x @ self.antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)

        return x

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


class ADGN(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        edge_dim: int = 0,
        num_layers: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        node_level_task: bool = False,
        activ_fun: str = "tanh",
        graph_conv: str = "NaiveAggr",
        bias: bool = True,
        train_weights: bool = True,
        weight_sharing: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.gamma = gamma
        self.activ_fun = activ_fun
        self.bias = bias
        self.train_weights = train_weights
        self.weight_sharing = weight_sharing
        self.emb = Linear(self.input_dim, self.hidden_dim)
        
        self.convs = ModuleList()
        for _ in range(1 if self.weight_sharing else self.num_layers):
            self.convs.append(
                AntiSymmetricConv(
                    in_channels=self.hidden_dim,
                    edge_channels = self.edge_dim,
                    num_iters=self.num_layers if weight_sharing else 1,
                    gamma=self.gamma,
                    epsilon=self.epsilon,
                    activ_fun=self.activ_fun,
                    graph_conv=graph_conv,
                    bias=self.bias,
                )
            )

        if not train_weights:
            # for param in self.enc.parameters():
            #    param.requires_grad = False
            for param in self.convs.parameters():
                param.requires_grad = False

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
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        x = self.emb(x) if self.emb else x
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)

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
    

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_') and k != 'convs' and k != 'emb' and k != 'readout')
        return f"ADGN({params})"

if __name__ == "__main__":
    hidden_dim = 65
    num_layers = 28
    input_dim = 1
    output_dim = 1

    model = ADGN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        
        node_level_task=True  # or False, as needed
    )
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")