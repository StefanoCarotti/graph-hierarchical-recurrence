import os
import torch
import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_tar
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from tqdm import tqdm

import time


urls = {
    'charge_train': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/charge/train_data.pt',
    'charge_val': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/charge/val_data.pt',
    'charge_test': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/charge/test_data.pt',

    'energy_train': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/energy/train_data.pt',
    'energy_val': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/energy/val_data.pt',
    'energy_test': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/energy/test_data.pt',

    'synth_train': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-synth/train_data.pt',
    'synth_val': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-synth/val_data.pt',
    'synth_test': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-synth/test_data.pt',
}


NODE_LVL_TASKS = ['sssp', 'ecc', 'charge']
GRAPH_LVL_TASKS = ['diam', 'energy']
TASKS = NODE_LVL_TASKS + GRAPH_LVL_TASKS


def constant_features_transform(data: Data, value: torch.float) -> Data: #type: ignore
    """Set all node features to a constant value. Required by reviewer suggestion."""
    data.x[:, 1] = torch.full_like(data.x[:,1], value) #type: ignore
    return data


ConstantFeatTransform = lambda value: lambda data: constant_features_transform(data, value)

class ECHO_Dataset(InMemoryDataset):
    # url = 'https://github.com/gravins/Anti-SymmetricDGN/raw/refs/heads/main/graph_prop_pred/data.tar.gz'

    def __init__(self, root, name, split='train', pre_transform=None, transform=None, dataset_path=None, max_ecc=None, max_diam=None, max_sssp=None, max_charge=None, k=None, constant_feature=None, **kwargs):
        """"
        Args:
            dataset_path (str): Path to the dataset excluding the split. The split will be added automatically, e.g. if the dataset_path is like 'data/chembl_syntetic_train.pt' then write 'data/chembl_syntetic'
        """

        self.dataset_path = dataset_path
        assert name in TASKS, f'{name} is not in {TASKS}'
        assert split in ['train', 'val', 'test']

        self.split = split
        self.name = name
        self.max_ecc = max_ecc
        self.max_diam = max_diam
        self.max_sssp = max_sssp
        self.max_charge = max_charge
        self.constant_feature = constant_feature
        self.pre_transform = pre_transform
        self.k = k

        if self.constant_feature is not None: 
            print(f'Using constant features with value {self.constant_feature}')
            self.pre_transform = ConstantFeatTransform(self.constant_feature)
        else:
            self.pre_transform = pre_transform

        print(f'Initializing ECHO_Dataset with name: {name}, split: {split}, constant_feature: {self.constant_feature}')
        print(f'Pre-transform: {self.pre_transform}')

        super().__init__(root, pre_transform=self.pre_transform, transform=transform, **kwargs)
        self.data, self.slices, self.max_ecc, self.max_diam, self.max_sssp, self.max_charge = torch.load(self.processed_paths[0], weights_only=False)
        self.scaling_factor = {
            'sssp': self.max_sssp,
            'ecc': self.max_ecc,
            'diam': self.max_diam,
            'charge': self.max_charge,
            'energy': self.max_charge
        }


    @property
    def num_classes(self) -> int:
        return 1

    @property
    def num_features(self) -> int:
        return self.data.x.shape[1]
    
    @property
    def is_node_level_task(self) -> bool:
        return self.name in NODE_LVL_TASKS
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    

    @property
    def processed_file_names(self):
        if self.constant_feature is not None:
            return [f"{self.split}_{self.name}_const_feat_{self.constant_feature}.pt"]
        return ["{}_{}{}{}.pt".format(
            self.split, 
            self.name, 
            "_khop" if self.pre_transform is not None else "", 
            "_" + str(self.k) if self.pre_transform is not None else ""
        )] 
    
    @property
    def processed_paths(self):
        return [osp.join(self.processed_dir, file_name) for file_name in self.processed_file_names]



    @property 
    def raw_file_names(self):
        if self.dataset_path is not None:
            return [f'{self.dataset_path}_{self.split}_data.pt']
        else:
            return [f'{self.split}_data.pt']
    # def download(self):
    #     targz = download_url(self.url, self.raw_dir)
    #     extract_tar(targz, self.raw_dir)
    #     os.unlink(targz)

    def download(self):
        print(f'Downloading {self.name} data for split {self.split}')
        print(f'Raw dir: {self.raw_dir}')
        if self.name == 'charge':
            for split in ['train', 'val', 'test']:
                url = urls[f'charge_{split}']
                download_url(url, self.raw_dir)
        elif self.name == 'energy':
            for split in ['train', 'val', 'test']:
                url = urls[f'energy_{split}']
                download_url(url, self.raw_dir)
        else:
            for split in ['train', 'val', 'test']:
                url = urls[f'synth_{split}']
                download_url(url, self.raw_dir)

    def normalize(self, data_list):
        # normalize labels
        # max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[0]
        # max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]
        # for dset in node_labels.keys():
        #     node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]
        #     graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]
        print(f'working on split {self.split}')
        print(self.max_diam, self.max_ecc, self.max_sssp)
        if self.max_diam is None or self.max_ecc is None or self.max_sssp is None or self.max_charge is None:
            print(f'Calculating max values for {self.split}')
            self.max_diam = 0
            self.max_ecc = 0
            self.max_sssp = 0
            self.max_charge = 0
            for data in data_list:
                if self.name == 'charge':
                    self.max_charge = max(self.max_charge, abs(data.y[:,0]).max().item())
                elif self.name == 'energy':
                    self.max_charge = max(self.max_charge, abs(data.y[:,0]).max().item())
                else:
                    self.max_ecc  = max(self.max_ecc, data.y[:,0].max().item())
                    self.max_diam = max(self.max_diam, data.y[:,1].max().item())
                    self.max_sssp = max(self.max_sssp, data.y[:,2].max().item())
            
        print(self.max_diam, self.max_ecc, self.max_sssp, self.max_charge)

        if self.name in ['diam', 'ecc', 'sssp']:
            for data in data_list:

                data.y[:, 0] /= self.max_ecc
                data.y[:, 1] /= self.max_diam
                data.y[:, 2] /= self.max_sssp

        return data_list


    def process(self):
        pre_process_times = []

        if self.name in ['charge', 'energy'] and self.constant_feature is not None:
            raise ValueError('Constant features not supported for charge or energy or ecc tasks')
        
        if self.name == 'charge':
            data_list = torch.load(f'{self.root}/charge/raw/{self.split}_data.pt', weights_only=False)
        elif self.name == 'energy':
            data_list = torch.load(f'{self.root}/energy/raw/{self.split}_data.pt', weights_only=False)

        else:
            fname = f'{self.root}/{self.name}/raw/{self.split}_data.pt'
            print(f'Loading {fname}')
            data_list = torch.load(fname, weights_only=False)

        if self.name != 'charge' and self.name != 'energy':
            print(f'Normalizing {self.name} data')
            tmp = data_list[0].y.clone()
            data_list = self.normalize(data_list)
            assert not data_list[0].y.allclose(tmp)

        for i, data in tqdm(enumerate(data_list), total=len(data_list), desc=f'Processing {self.name} data'):
            
            if self.name == 'diam':
                data_list[i].y = data.y[:, 1][0]
            elif self.name == 'ecc':    
                data_list[i].y = data.y[:, 0]
            elif self.name == 'sssp':
                data_list[i].y = data.y[:, 2]
            elif self.name == 'charge':
                data_list[i].y = data.y[:, 0]
            elif self.name == 'energy':
                data_list[i].y = data.y[:, 0]
            
            data_list[i].x = torch.tensor(data.x).float()

        
            
            if self.pre_transform is not None:
                start = time.time()
                # convert every tensor in data_list[i] from numpy to torch tensor
                    
                data_list[i] = self.pre_transform(data_list[i])
                end = time.time()
                pre_process_times.append(end - start)

        for d in data_list:
            d.x = torch.Tensor(d.x).float().squeeze()
            d.y = d.y.float()

    
        data, slices = self.collate(data_list)
        times = torch.tensor(pre_process_times)
        print(f'\033[91mPreprocessing times: {times.sum().item()}, task: {self.name}, split: {self.split}\033[0m')
        torch.save((data, slices, self.max_ecc, self.max_diam, self.max_sssp, self.max_charge), self.processed_paths[0])
