from .echo_dataset import ECHO_Dataset, TASKS
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.utils import from_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
import torch
import numpy as np
from scipy import sparse
import pandas as pd
from tqdm import tqdm
import networkx as nx
import time


def safe_convert(val):
    if pd.isna(val):
        return val  # Leave NaN/None as-is

    if isinstance(val, str):
        val_stripped = val.strip()
        lower_val = val_stripped.lower()

        # Handle booleans
        if lower_val == 'true':
            return True
        elif lower_val == 'false':
            return False

        # Try to convert to int (only if it looks like an integer)
        try:
            int_val = int(val_stripped)
            if str(int_val) == val_stripped or str(int_val) == val_stripped.lstrip('0'):
                return int_val
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(val_stripped)
        except ValueError:
            return val_stripped  # Return stripped string

    # For non-string types
    if isinstance(val, (int, float, bool)):
        return val

    try:
        return float(val)
    except (ValueError, TypeError):
        return val

def khop_transform(khop: int, data: Data) -> Data:
    A = to_dense_adj(data.edge_index)
    #make data.x a torch tensor
    if not isinstance(data.x, torch.Tensor):
        data.x = torch.tensor(data.x, dtype=torch.float)
    dist = floyd_warshall(
        A.squeeze().cpu().numpy(), directed=True, unweighted=True
    )
    dist = np.where(np.isfinite(dist), dist, -1).astype(
        np.int32
    )  # -1s are nodes in same batch, different graph

    k_edge_index = torch.LongTensor()  # int64
    k_idx = torch.ByteTensor()  # int8
    idx = [0]
    data.max_k = [np.max(dist)]

    for k in range(1, min(np.max(dist), khop) + 1):
        A_k_hop = (dist == k).astype(int)
        k_edges = from_scipy_sparse_matrix(sparse.csr_matrix(A_k_hop))[0]
        k_edge_index = torch.cat((k_edge_index, k_edges), dim=1)
        idx.append(k_edge_index.shape[1])
        k_idx = torch.cat((k_idx, k * torch.ones(k_edges.shape[1])))

    idx = torch.tensor(idx, dtype=torch.int32)
    data.k_idx = k_idx.to(device=data.x.device)
    data.k_edge_index = k_edge_index.to(device=data.x.device)

    # Check labels are as expected and k-hop indices match edge_index for k=1 (accounting for rogue self connections)
    # TODO: can drop this check in later versions
    num_self_connections = ((data.edge_index[0] - data.edge_index[1]) == 0).sum()
    assert (data.k_edge_index[:, data.k_idx == 1].shape[1]) == (
        data.edge_index.shape[1] - num_self_connections
    )
    for k in range(1, min(np.max(dist), khop) + 1):
        assert torch.equal(
            data.k_edge_index[:, data.k_idx == k],
            data.k_edge_index[:, idx[k - 1] : idx[k].item()],
        )

    return data

KHopTransform = lambda k: lambda data: khop_transform(k, data)


def constant_features_transform(data: Data, value: torch.float) -> Data:
    """Set all node features to a constant value. Required by reviewer suggestion."""
    num_nodes = data.num_nodes
    data.x = torch.full((num_nodes, 1), value, device=data.x.device)
    return data


ConstantFeatTransform = lambda value: lambda data: constant_features_transform(data, value)



def get_dataset(root: str, task=None, pre_transform=None, force_reload=False, **kwargs):
    """
    Loads the ECHO_Dataset dataset for a specific task.

    This function initializes ECHO_Dataset objects for train, validation, and test splits.
    If the dataset files are not present in the specified `root` directory, they will be 
    automatically downloaded from Hugging Face repositories as defined in the ECHO_Dataset class.

    Args:
        root (str): Root directory where the dataset should be saved.
        task (str, optional): The name of the task (e.g., 'charge', 'energy', 'sssp', 'ecc', 'diam').
        pre_transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before every access.
        force_reload (bool, optional): Whether to re-process the dataset.
        **kwargs: Additional arguments passed to ECHO_Dataset(e.g., k).

    Returns:
        tuple: A tuple containing (data_train, data_valid, data_test, num_features, num_classes).
    """

    assert task is None or task in TASKS
    k = kwargs.get('k', None)

    constant_feature = kwargs.get('constant_feature', None)
    if constant_feature is not None:
        print("Applying constant feature transform with value:", constant_feature)
        pre_transform = ConstantFeatTransform(constant_feature)
    
    data_train = ECHO_Dataset(
        root, name=task, split='train', pre_transform=pre_transform, force_reload=force_reload, k=k,constant_feature=constant_feature) 
    data_valid = ECHO_Dataset(
        root, name=task, split='val',   pre_transform=pre_transform, force_reload=force_reload, max_diam=data_train.max_diam, max_ecc=data_train.max_ecc, max_sssp=data_train.max_sssp, k=k,constant_feature=constant_feature)
    data_test = ECHO_Dataset(
        root, name=task, split='test',   pre_transform=pre_transform, force_reload=force_reload, max_diam=data_train.max_diam, max_ecc=data_train.max_ecc, max_sssp=data_train.max_sssp, k=k, constant_feature=constant_feature)
    
    num_features = data_train.num_features
    num_classes = data_train.num_classes

    return data_train, data_valid, data_test, num_features, num_classes


def convert_to_csv_like(data_list):
    """
    Converts a list of PyTorch Geometric data objects to a tabular (CSV-like) format.

    Args:
        data_list (list): List of PyTorch Geometric data objects.

    Returns:
        pd.DataFrame: A DataFrame containing the tabular representation of the data.
    """
    rows = []
    for data in tqdm(data_list):
        row = {
            'y': data.y.item() if data.y.numel() == 1 else data.y.tolist(),
            'edge_index': data.edge_index.tolist(),
            'x': data.x.tolist(),
            'edge_attr': data.edge_attr.tolist() if data.edge_attr is not None else None,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def convert_to_graphs(dataframe):
    """
    Converts a DataFrame containing graph data into a list of torch_geometric.Data objects.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing graph data with columns 'y', 'edge_index', 'x', and optionally 'g_type'.

    Returns:
        list: A list of torch_geometric.Data objects.
    """
    graphs = []
    for _, row in dataframe.iterrows():
        graph = Data(
            x=torch.tensor(row['x'], dtype=torch.float),
            edge_index=torch.tensor(row['edge_index'], dtype=torch.long),
            y=torch.tensor(row['y'], dtype=torch.float),
            edge_attr=torch.tensor(row['edge_attr'], dtype=torch.float) if row['edge_attr'] is not None else None,
        )

        graphs.append(graph)
    return graphs

def compute_graph_diameter(g):
    return nx.diameter(to_networkx(g))

def compute_mae(predictions: list, scaling_factor: float = 1.0) -> float:
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values.
    """
    y_true = torch.cat([pred['y'] for pred in predictions]).flatten()
    y_pred = torch.cat([pred['y_p'] for pred in predictions]).flatten()
    mae = torch.mean(torch.abs(y_true * scaling_factor - y_pred * scaling_factor))
    return mae.item()

DATA = ['GraphProp']