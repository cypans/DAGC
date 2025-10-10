# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 13:32
@Auth: Panchuying
@File: main.py
@IDE: PyCharm
@Theme:
"""
from torch import optim, nn
from utils.run import run_one_time, run_multiple_times
from utils.args import args
from data_process.dataset import load_cluster_data
from model.net import GNN
from utils.utils import set_seed, printf, save_results

# if __name__ == '__main__':
#     for seed in range(400):
#         set_seed(seed)
#         printf(f"seed:{seed}------------------------------------------------")
#         loader, dataset = load_cluster_data(args.data_dir, args.dataset_name, args.split_type, args.num_parts,
#                                             args.batch_size)
#         model = GNN(in_channels=1, out_channels=1, node_feat_dim=dataset.graph['node_feat'].shape[-1], hidden_feat_din=args.hidden_feat_dim,
#                     out_feat_dim=dataset.num_classes, kernel_size=[args.kernel_size1, args.kernel_size2], leaf=args.leaf).to(args.device)
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#         criterion = nn.CrossEntropyLoss()
#         # run_one_time(model, loader, dataset, optimizer, criterion, args)
#         run_multiple_times(model, loader, dataset, optimizer, criterion, args, args.times)
#
#     save_results(args, 0000000)

if __name__ == '__main__':
    set_seed(2025)
    loader, dataset = load_cluster_data(args.data_dir, args.dataset_name, args.split_type, args.num_parts,
                                        args.batch_size)
    model = GNN(in_channels=1, out_channels=1, node_feat_dim=dataset.graph['node_feat'].shape[-1], hidden_feat_din=args.hidden_feat_dim,
                out_feat_dim=dataset.num_classes, kernel_size=[args.kernel_size1, args.kernel_size2], leaf=args.leaf).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    run_one_time(model, loader, dataset, optimizer, criterion, args)
    # run_multiple_times(model, loader, dataset, optimizer, criterion, args, args.times)