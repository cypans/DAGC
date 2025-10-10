# -*- coding:utf-8 -*-
"""
@Time: 2024/12/23 9:02
@Auth: Panchuying
@File: dataset.py
@IDE: PyCharm
@Theme:
"""
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, Planetoid, Actor, WebKB, WikipediaNetwork
from ogb.nodeproppred import NodePropPredDataset

import numpy as np
import scipy.sparse as sp
from os import path
#from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown
import scipy
from data_process.data_utils import *
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch.utils.data import Dataset
from utils.args import args


class NCDataset(Dataset):#object
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', num_nodes=None):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        num_nodes: 节点总数，用于生成 mask
        """
        if num_nodes is None:
            num_nodes = self.graph['num_nodes']  # 确保能获取节点数

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=.5, valid_prop=.25, ignore_negative=ignore_negative)

            # 初始化全 False 的掩码
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            # 将对应索引的位置设置为 True
            train_mask[train_idx] = True
            val_mask[valid_idx] = True
            test_mask[test_idx] = True

            split_mask = {'train_mask': train_mask,
                          'val_mask': val_mask,
                          'test_mask': test_mask}
            return split_mask

        elif split_type == 'class_rand':
            split_idx = class_rand_splits(
                self.label, label_num_per_class=20, valid_num=500, test_num=1000)


            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[split_idx['train']] = True
            val_mask[split_idx['valid']] = True
            test_mask[split_idx['test']] = True

            split_mask = {'train_mask': train_mask,
                          'val_mask': val_mask,
                          'test_mask': test_mask}
            return split_mask

        elif split_type == 'fixed_splits':
            if self.name in ["Cora", "CiteSeer", "PubMed"]:
                dataset = Planetoid(root=args.data_dir, name=self.name)
                data = dataset[0]  # 获取第一个图数据
                split_mask = {'train_mask': data.train_mask,
                              'val_mask': data.val_mask,
                              'test_mask': data.test_mask}

                return split_mask
            elif self.name in ["actor", 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel']:
                return self.load_fixed_splits()

            elif self.name in ["ogbn-products", 'ogbn-arxiv', 'ogbn-proteins']:
                return self.load_fixed_splits()

            else:
                if self.name in ["amazon-computer", "amazon-photo", 'coauthor-physics', "coauthor-cs"]:
                    split_idx = load_fixed_splits(
                        data_dir=args.data_dir+'/Amazon', name=self.name)
                else:
                    split_idx = load_fixed_splits(
                        data_dir=args.data_dir, name=self.name)
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)

                train_mask[split_idx['train']] = True
                val_mask[split_idx['valid']] = True
                test_mask[split_idx['test']] = True

                split_mask = {'train_mask': train_mask,
                              'val_mask': val_mask,
                              'test_mask': test_mask}
                return split_mask


    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

    def to(self, device):
        """
        将所有数据迁移到指定设备上（例如 GPU）。
        :param device: 'cuda' or 'cpu'
        """
        # 遍历所有属性
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                # 如果属性是张量，则迁移到指定设备
                self.__dict__[key] = value.to(device)
            elif isinstance(value, dict):
                # 如果属性是字典，继续遍历字典中的每一项
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        value[sub_key] = sub_value.to(device)

def load_dataset(data_dir, dataname, split_type, sub_dataname=''):
    """ Loader for NCDataset
        Returns NCDataset
    """
    print(dataname)
    if dataname in ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname, split_type)
    elif dataname in ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname, split_type)
    elif dataname in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_hetero_dataset(data_dir, dataname, split_type)
    elif dataname == 'wikics':
        dataset = load_wikics_dataset(data_dir, split_type)
    elif dataname in ('ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'):
        dataset = load_ogb_dataset(data_dir, dataname, split_type)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir, split_type)
    elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(data_dir, dataname, split_type)
    # elif dataname in ('chameleon', 'squirrel'):
    #     dataset = load_wiki_new(data_dir, dataname, split_type)
    elif dataname in ('actor'):
        dataset = load_actor_dataset(data_dir, dataname, split_type)
    elif dataname in ('Cornell', 'Texas', 'Wisconsin'):
        dataset = load_webkb_dataset(data_dir, dataname, split_type)
    elif dataname in ('Chameleon', 'Squirrel'):
        dataset = load_wikipedianetwork_dataset(data_dir, dataname, split_type)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_planetoid_dataset(data_dir, name, split_type, no_feat_norm=True):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{data_dir}',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{data_dir}', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset


def load_wiki_new(data_dir, name, split_type):
    path= f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
    data=np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat=data['node_features'] # unnormalized
    labels=data['node_labels']
    edges=data['edges'] #(E, 2)
    edge_index=edges.T

    dataset = NCDataset(name)

    edge_index=torch.as_tensor(edge_index)
    node_feat=torch.as_tensor(node_feat)
    labels=torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels
    dataset.num_classes = torch.unique(labels).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset

def load_wikics_dataset(data_dir,split_type):
    wikics_dataset = WikiCS(root=f'{data_dir}/wikics/')
    data = wikics_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset('wikics')
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset

def load_hetero_dataset(data_dir, name, split_type):
    #transform = T.NormalizeFeatures()
    torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir+'/HeterophilousGraphDataset')
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    ## dataset splits are implemented in data_utils.py
    '''
    dataset.train_idx = torch.where(data.train_mask[:,0])[0]
    dataset.valid_idx = torch.where(data.val_mask[:,0])[0]
    dataset.test_idx = torch.where(data.test_mask[:,0])[0]
    '''

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset


def load_amazon_dataset(data_dir, name, split_type):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}/Amazon',
                                 name='Photo', transform=transform)
    elif name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}/Amazon',
                                 name='Computers', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset

def load_coauthor_dataset(data_dir, name, split_type):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}/Coauthor',
                                 name='CS', transform=transform)
    elif name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}/Coauthor',
                                 name='Physics', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset

def load_ogb_dataset(data_dir, name, split_type):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        train_mask = torch.zeros(dataset.graph['num_nodes'], dtype=torch.bool)
        val_mask = torch.zeros(dataset.graph['num_nodes'], dtype=torch.bool)
        test_mask = torch.zeros(dataset.graph['num_nodes'], dtype=torch.bool)

        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

        split_mask = {'train_mask': train_mask,
                      'val_mask': val_mask,
                      'test_mask': test_mask}
        return split_mask
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1)
    dataset.num_classes = torch.unique(dataset.label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset

def load_pokec_mat(data_dir, split_type):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    dataset.num_classes = torch.unique(dataset.label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset


def load_actor_dataset(data_dir, name, split_type):
    transform = T.NormalizeFeatures()
    torch_dataset = Actor(root=f'{data_dir}/Actor', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    def get_fixed_splits():
        split_mask = {'train_mask': data.train_mask,
                      'val_mask': data.val_mask,
                      'test_mask': data.test_mask}
        return split_mask
    dataset.load_fixed_splits = get_fixed_splits
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset


def load_webkb_dataset(data_dir, name, split_type):
    transform = T.NormalizeFeatures()
    torch_dataset = WebKB(root=f'{data_dir}/WebKB', name=name, transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    def get_fixed_splits():
        split_mask = {'train_mask': data.train_mask,
                      'val_mask': data.val_mask,
                      'test_mask': data.test_mask}
        return split_mask
    dataset.load_fixed_splits = get_fixed_splits
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset


def load_wikipedianetwork_dataset(data_dir, name, split_type):
    transform = T.NormalizeFeatures()
    torch_dataset = WikipediaNetwork(root=f'{data_dir}/WikipediaNetwork', name=name, transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    def get_fixed_splits():
        split_mask = {'train_mask': data.train_mask,
                      'val_mask': data.val_mask,
                      'test_mask': data.test_mask}
        return split_mask
    dataset.load_fixed_splits = get_fixed_splits
    dataset.label = label
    dataset.num_classes = torch.unique(label).numel()
    dataset.graph = dataset.graph | dataset.get_idx_split(split_type=split_type)

    return dataset


def convert_to_pyg_data(dataset):
    """
    将自定义的 NCDataset 转换为 PyTorch Geometric 的 Data 格式
    """
    graph = dataset.graph
    data = Data(
        x=graph['node_feat'],                # 节点特征
        edge_index=graph['edge_index'],      # 边索引
        y=dataset.label,                      # 节点标签
        edge_attr=graph['edge_feat'] if 'edge_feat' in graph else None,  # 边属性
        train_mask=graph['train_mask'] if 'train_mask' in graph else None,
        val_mask=graph['val_mask'] if 'val_mask' in graph else None,
        test_mask=graph['test_mask'] if 'test_mask' in graph else None
    )
    return data

def load_cluster_data(data_dir, dataset_name, split_type, num_parts, batch_size):
    """
    封装成 ClusterData 和 ClusterLoader
    :param data_dir: 数据存储路径
    :param dataset_name: 数据集名称
    :param split_type: 划分数据集的类型 'random' 'class_rand' 'fixed_splits'
    :param num_parts: 将整图划分成的子图簇数量
    :param batch_size: 每次加载的子图簇数量
    """
    # 加载自定义的 NCDataset
    dataset = load_dataset(data_dir, dataset_name, split_type=split_type)
    # dataset.to(device)

    # 转换为 PyG 格式的数据
    pyg_data = convert_to_pyg_data(dataset)

    # 使用 ClusterData 进行子图簇划分
    cluster_data = ClusterData(pyg_data, num_parts=num_parts, recursive=False, )

    # 使用 ClusterLoader 批量加载子图簇
    loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True, num_workers=5)

    return loader, dataset

if __name__ == '__main__':
    # Load a dataset
    data_dir = '../data'
    dataset_name = 'amazon-photo'   # ['amazon-photo','Cora']
    split_type = "fixed_splits"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_parts = 10
    batch_size = 1
    loader, dataset = load_cluster_data(data_dir, dataset_name, split_type,
                                        num_parts=num_parts, batch_size=batch_size)
    for subgraph_data in loader:
        subgraph_data = subgraph_data.to(device)
        print(subgraph_data)

