import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from huggingface_hub import hf_hub_download
from torch_geometric.utils import to_undirected
import random

class LRIM(InMemoryDataset):
    """
    Dataset class for Ising model data following PyG InMemoryDataset format.
   
    Args:
        root (str): Root directory where the dataset should be saved.
        split (str): Which split to use ('train', 'val', 'test').
        task (str): Task to perform.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
    """
    def __init__(self, root, name, hf_repo='jmathys/lrim_graph_benchmark', pre_transform=None, transform=None):
        self.version = repr(pre_transform)
        self.name = name
        self.hf_repo = hf_repo

        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        print(f'Loaded {self.processed_paths[0]}')

    @property
    def raw_file_names(self):
        return [f'{self.name}.pt']

    @property
    def processed_file_names(self):
        return [f'{self.name}_{self.version}_data.pt']

    def download(self):
        """
        Download dataset from HuggingFace if not present locally.
        The filename in HF repo should match the dataset name (e.g., 'lrim_16_0.6_1k.pt').
        """
        dataset_path = os.path.join(self.raw_dir, f'{self.name}.pt')

        if os.path.exists(dataset_path):
            print(f"Dataset already exists at {dataset_path}")
            return

        if self.hf_repo is None:
            raise FileNotFoundError(
                f"Dataset file not found at {dataset_path} and hf_repo is None. "
                f"Please provide the dataset file or set hf_repo to download from HuggingFace."
            )

        # The name directly specifies which file to download from HF
        # e.g., name='lrim_16_0.6_1k' downloads 'lrim_16_0.6_1k.pt' from the repo
        print(f"Downloading {self.name}.pt from HuggingFace repository: {self.hf_repo}")
        try:
            hf_hub_download(
                repo_id=self.hf_repo,
                filename=f'{self.name}.pt',
                repo_type='dataset',
                local_dir=self.raw_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded to {dataset_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {self.name}.pt from HuggingFace. "
                f"Make sure the file exists in repo {self.hf_repo}. Error: {e}"
            )

    def process(self):
        """
        Loads the entire dataset from a .pt file and splits based on fixed percentages.
        """
        dataset_path = os.path.join(self.raw_dir, f'{self.name}.pt')
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found at {dataset_path}. "
                f"The download() method should have been called automatically."
            )
        
        print(f"Loading dataset from {dataset_path}")
        full_dataset = torch.load(dataset_path, weights_only=False)
        random.seed(42)  
        full_dataset = random.sample(full_dataset, 1000)   
        processed_data_list = [] # <--- Create a new list to hold processed data
            
        for data in full_dataset:
            # 1. Standardize types
            data.x = data.x.float()
            data.edge_index = data.edge_index.long()
            data.y = data.y.float()

            if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
                raise ValueError("Data object must have .x, .edge_index and .y attributes")
            
            data.edge_index = to_undirected(data.edge_index)

            # 3. Strip original excess attributes
            for attr in list(data.keys()):
                if attr not in ['x', 'edge_index', 'y', 'edge_attr']:
                    del data[attr]

            # 4. === YOUR CRITICAL ADDITION ===
            # Apply Graclus pre_transform AFTER they strip the original garbage, 
            # so our new hierarchical attributes don't get deleted!
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            processed_data_list.append(data)

        self.data_size = len(processed_data_list)
        print(f"Total dataset size: {self.data_size} samples")
        
        # Collate the data into a single batch
        data, slices = self.collate(processed_data_list)
        
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved processed data to {self.processed_paths[0]}")

    def get_idx_split(self):
        total_size = len(self) # Use the actual number of graphs loaded

        # 80% train, 10% val, 10% test
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        indices = torch.randperm(total_size)
        splits = {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }

        return splits

        return splits
