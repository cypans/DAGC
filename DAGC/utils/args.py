# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 13:35
@Auth: Panchuying
@File: args.py
@IDE: PyCharm
@Theme:
"""
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Decoupled Multi-Hop Neighbor Aggregation Graph Neural Network")

    # Positional arguments
    parser.add_argument('--dataset_name', type=str, default='Cora', help='Choose a dataset from [Cora, CiteSeer, PubMed]')
    parser.add_argument('--data_dir', type=str, default='./data', help='the data_process path of the big data_process')
    parser.add_argument('--split_type', type=str, default='fixed_splits', choices=['fixed_splits', 'class_rand', 'random'], help='the split type of dataset to train, validation and test')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda:0', 'cuda:1'], help='Device to use for training')
    parser.add_argument('--epoch', type=int, default=2500, help='Number of epochs to train')
    parser.add_argument('--num_parts', type=int, default=5, help='when using clusterloader to train, the number of split parts')
    parser.add_argument('--batch_size', type=int, default=1, help='when using clusterloader to train, batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--kernel_size1', type=int, default=4, help='the kernel size')
    parser.add_argument('--kernel_size2', type=int, default=3, help='the kernel size')
    parser.add_argument('--hidden_feat_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--times', type=int, default=10, help='the times to run, then calculate teh avg and std')
    parser.add_argument('--leaf', type=float, default=3, help='tricks')
    parser.add_argument('--model_path', type=str, default=None, help='model path')

    return parser.parse_args()


def get_args():
    return args


# #other_file.py
# from main import get_args或者from main import GLOBAL_ARGS
args = parse_args()  # 其实放到这个位置就是全局变量
# GLOBAL_ARGS = args