import torch
import h3
import os.path as osp
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.datasets import CityNetwork
from torch_geometric.transforms import RandomNodeSplit

class CityNetworkH3(CityNetwork):
    def __init__(self, root, name, resolution=8, minimalist=True, regression=True, split_props=(0.6, 0.2, 0.2), **kwargs):
        """
        Args:
            resolution (int): H3 resolution (8 or 9 recommended).
            minimalist (bool): If True, throws away 35 junk features, keeping only (x, y) and Edge Length.
            regression (bool): If True, loads raw continuous eccentricities for MSE training.
        """
        self.resolution = resolution
        super().__init__(root, name, **kwargs)
        # Handle PyG version differences
        if hasattr(self, '_data'):
            data = self._data
        else:
            data = self.data

        # ---------------------------------------------------------
        # STEP 1: LOAD MISSING EDGE FEATURES (Physical Distance)
        # ---------------------------------------------------------
        if data.edge_attr is None:
            raw_file_path = osp.join(self.raw_dir, self.name, "edge_features.pt")
            if osp.exists(raw_file_path):
                print(f"[{self.name}] Loading missing edge attributes...")
                data.edge_attr = torch.load(raw_file_path)
            else:
                print(f"[{self.name}] WARNING: edge_features.pt not found.")

        # PURIFICATION: Slice Edge Features
        if data.edge_attr is not None and minimalist:
            print(f"[{self.name}] PURIFICATION: Keeping only Edge Length. Dropping Speed/Type.")
            # Keep only column 0 (Length in meters)
            data.edge_attr = data.edge_attr[:, 0:1]

        # ---------------------------------------------------------
        # STEP 2: LOAD & NORMALIZE TARGETS
        # ---------------------------------------------------------
        if regression:
            target_path = osp.join(self.raw_dir, self.name, "16-hop_eccentricities.pt")
            if osp.exists(target_path):
                print(f"[{self.name}] Loading continuous targets...")
                # 1. Load Raw Meters (Needed for MAE metric)
                data.y_raw = torch.load(target_path)
                # 2. Compute Stats (Global normalization for Transductive task)
                y_mean = data.y_raw.mean()
                y_std = data.y_raw.std()
                # 3. Create Normalized Target (Needed for Training Loss)
                data.y_norm = (data.y_raw - y_mean) / (y_std + 1e-6)
                # 4. Save Stats for De-normalization later
                data.y_mean = y_mean
                data.y_std = y_std
                print(f" -> Target Mean: {y_mean:.1f}m | Std: {y_std:.1f}m")
            else:
                print(f"[{self.name}] WARNING: '16-hop_eccentricities.pt' not found.")

        # ---------------------------------------------------------
        # STEP 3: PRESERVE RAW POSITIONS
        # ---------------------------------------------------------
        # Save raw Lat/Lon for H3 hashing later.
        # Crucial: Must be done BEFORE normalization.
        if not hasattr(data, 'pos') or data.pos is None:
            data.pos = data.x[:, 0:2].clone()

        # ---------------------------------------------------------
        # STEP 4: SCIENTIFIC PURIFICATION (Node Features)
        # ---------------------------------------------------------
        if minimalist:
            print(f"[{self.name}] PURIFICATION: Discarding 35 junk features. Keeping only x, y.")
            # Keep only col 0 (x) and 1 (y).
            data.x = data.x[:, 0:2]
        # ---------------------------------------------------------
        # STEP 5: NORMALIZE (CRITICAL FIX)
        # ---------------------------------------------------------
        print(f"[{self.name}] Normalizing inputs (Mean/Std)...")
        print(f"[{self.name}] Normalizing inputs (Mean/Std)...")
        # A. Node Normalization
        # Simple Logic: Just normalize whatever columns we have.
        mean_x = data.x.mean(dim=0)
        std_x = data.x.std(dim=0)
        data.x = (data.x - mean_x) / (std_x + 1e-6)
        data.x = torch.zeros((data.num_nodes, 2), dtype=torch.float)
        data.x[:, 1] = 1.0


        # B. Edge Normalization
        if data.edge_attr is not None:
            # Simple Logic: Just normalize whatever columns we have (Length).
            mean_e = data.edge_attr.mean(dim=0)
            std_e = data.edge_attr.std(dim=0)
            data.edge_attr = (data.edge_attr - mean_e) / (std_e + 1e-6)

        # ---------------------------------------------------------
        # STEP 6: COMPUTE H3 COARSE GRAPH
        # ---------------------------------------------------------
        data = self._compute_h3_graph(data)
        train_p, val_p, test_p = split_props
        # This transform automatically creates new boolean masks
        transform = RandomNodeSplit(
            split='train_rest',
            num_val=val_p,
            num_test=test_p
        )
        data = transform(data)

        # Save back to self
        if hasattr(self, '_data'):
            self._data = data
        else:
            self.data = data

    def _compute_h3_graph(self, data):
        print(f"[{self.name}] Generating H3 Coarse Graph (Res {self.resolution})...")
        pos_np = data.pos.numpy()
        try:
            h3_func = h3.latlng_to_cell
        except AttributeError:
            h3_func = h3.geo_to_h3

        h3_indices = [h3_func(lat, lon, self.resolution) for lon, lat in pos_np]
        unique_h3 = sorted(list(set(h3_indices)))
        h3_to_id = {h: i for i, h in enumerate(unique_h3)}
        cluster = torch.tensor([h3_to_id[h] for h in h3_indices], dtype=torch.long)
        num_coarse = len(unique_h3)

        try:
            h3_to_geo = h3.cell_to_latlng
        except AttributeError:
            h3_to_geo = h3.h3_to_geo
        coarse_centroids = [h3_to_geo(h) for h in unique_h3]
        coarse_pos = torch.tensor([(lon, lat) for lat, lon in coarse_centroids], dtype=torch.float)

        row, col = data.edge_index
        c_row, c_col = cluster[row], cluster[col]
        coarse_edge_index = torch.stack([c_row, c_col], dim=0)
        coarse_edge_index, _ = remove_self_loops(coarse_edge_index)
        coarse_edge_index = coalesce(coarse_edge_index, num_nodes=num_coarse)

        data.cluster = cluster
        data.num_coarse_nodes = num_coarse
        data.coarse_pos = coarse_pos
        data.coarse_edge_index = coarse_edge_index
        print(f" -> Coarse Nodes: {num_coarse} | Coarse Edges: {coarse_edge_index.size(1)}")
        return data