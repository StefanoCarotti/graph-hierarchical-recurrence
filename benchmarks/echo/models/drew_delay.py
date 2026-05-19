import torch
from torch import nn
from torch.nn import Linear, LeakyReLU, Sequential
from collections import OrderedDict
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    GCNConv,
    GINEConv
)


class DRew_GCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        num_layers=1,
        node_level_task=False,
        delay: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.delay = delay
        self.edges = kwargs.get("edges", None)

        self.emb = Linear(self.input_dim, self.hidden_dim)

        self.conv_func_name = "DRew_GCN"
        if delay:
            self.conv_func_name += "+delay"

        self.convs = nn.ModuleList()
        self.convlayer = GCNConv if self.edges is None else GINEConv
        for l in range(num_layers + 1):
            if self.edges is None:
                self.convs.append(
                    nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(l + 1)])
                )
            else:
                self.convs.append(
                    nn.ModuleList([
                        GINEConv(
                            nn = Linear(self.hidden_dim, self.hidden_dim),
                            edge_dim=2, 
                            train_eps=True
                    ) for _ in range(l + 1)])
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

    def forward(self, data):
        x, batch = data.x, data.batch
        k_edge_index, k_idx = data.k_edge_index, data.k_idx

        x = self.emb(x)
        x_l = [x]  # for previous layers' node features when using delay

        for l, layer_conv in enumerate(self.convs):
            for k in range(1, l + 2):
                edge_index_k_hop = k_edge_index[:, k_idx == k]
                conv = layer_conv[k - 1]  # separate weights across *khops*
                # Perform mean-aggregation over all of the khops
                if self.edges is None:
                    if k == 1:
                        x_k = conv(x, edge_index_k_hop) / k
                    elif self.delay:
                        x_k += conv(x_l[-k], edge_index_k_hop) / k
                    else:
                        x_k += conv(x, edge_index_k_hop) / k
                else:
                    if k == 1:
                        x_k = conv(x, edge_index_k_hop, edge_attr=data.edge_attr) / k
                    elif self.delay:
                        x_k += conv(x_l[-k], edge_index_k_hop, edge_attr=data.edge_attr) / k
                    else:
                        x_k += conv(x, edge_index_k_hop, edge_attr=data.edge_attr) / k

            x = torch.relu(x_k)

            if self.delay:
                x_l.append(x)

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



if __name__ == "__main__":
    hidden_dim = 78
    num_layers = 4
    input_dim = 1
    output_dim = 1

    model = DRew_GCN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        delay=False,
        node_level_task=True  # or False, as needed
    )
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")