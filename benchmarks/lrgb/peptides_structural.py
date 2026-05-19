import hashlib
import os.path as osp
import pickle
import shutil
import pandas as pd
import torch
from tqdm import tqdm

from torch_scatter import scatter_sum
from torch_geometric.nn import graclus
from torch_geometric.data import Data, download_url
from torch_geometric.data import InMemoryDataset

from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download

from ogb.utils import smiles2graph

class ClusterData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster':
            return self.num_coarse_nodes
        if key == 'coarse_edge_index':
            return self.num_coarse_nodes  
        return super().__inc__(key, value, *args, **kwargs)

def get_induced_coarse_edges(fine_edge_index, cluster, edge_attr):
    row, col = fine_edge_index
    cluster_row = cluster[row]
    cluster_col = cluster[col]
    
    mask = cluster_row != cluster_col
    coarse_edges_full = torch.stack([cluster_row[mask], cluster_col[mask]], dim=0)
    
    if coarse_edges_full.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    coarse_edge_index, inverse_indices = torch.unique(coarse_edges_full, dim=1, return_inverse=True)
    
    # We just count the bonds between the supernodes (bandwidth)
    ones = torch.ones(coarse_edges_full.size(1), dtype=torch.float)
    coarse_edge_attr = scatter_sum(ones, inverse_indices, dim=0).unsqueeze(1)
    
    return coarse_edge_index, coarse_edge_attr


def get_graclus_clusters_fixed_rounds(edge_index, num_nodes, rounds=1):
    current_edge_index = edge_index.clone() 
    final_cluster = torch.arange(num_nodes, dtype=torch.long)
    current_num_nodes = num_nodes
    
    for _ in range(rounds):
        cluster = graclus(current_edge_index, num_nodes=current_num_nodes)
        
        unique_c, cluster = torch.unique(cluster, return_inverse=True)
        num_clusters = len(unique_c)
        
        if num_clusters == current_num_nodes:
            break
            
        final_cluster = cluster[final_cluster]
        
        # We compute the induced edges for the NEXT round 
        dummy_edge_attr = torch.ones((current_edge_index.size(1), 1), dtype=torch.float)
        current_edge_index, _ = get_induced_coarse_edges(
            current_edge_index, cluster, dummy_edge_attr
        )
        current_num_nodes = num_clusters

    return final_cluster, current_num_nodes


class PeptidesStructuralDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.

        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-structural')

        self.url = 'https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1'
        self.version = '9786061a34298a0684150f2e4ff13f47'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return 'peptide_structure_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_structure_dataset.csv.gz'))
        smiles_list = data_df['smiles']
        target_names = ['Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                        'Inertia_valence_a', 'Inertia_valence_b',
                        'Inertia_valence_c', 'length_a', 'length_b', 'length_c',
                        'Spherocity', 'Plane_best_fit']
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0)

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            num_nodes = int(graph['num_nodes'])
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            
            # y is a pandas series, convert to list then tensor
            y_tensor = torch.Tensor([y.tolist()])

            # --- Extract Coarse Graph ---
            # Used rounds=1 to match the functional config
            cluster, num_coarse = get_graclus_clusters_fixed_rounds(edge_index, num_nodes, rounds=1)
            coarse_edge_index, coarse_edge_attr = get_induced_coarse_edges(edge_index, cluster, edge_attr)

            # Use our custom ClusterData instead of standard Data
            data = ClusterData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y_tensor,
                __num_nodes__=num_nodes,
                cluster=cluster,
                num_coarse_nodes=num_coarse,
                coarse_edge_index=coarse_edge_index,
                coarse_edge_attr=coarse_edge_attr
            )

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide_structure.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


if __name__ == '__main__':
    dataset = PeptidesStructuralDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())