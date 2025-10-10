# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 13:28
@Auth: Panchuying
@File: utils.py
@IDE: PyCharm
@Theme:
"""
import os
from datetime import datetime
import torch_sparse
import torch
import numpy as np
import random


log_entries = ""


def normalize_adj(adj: torch_sparse.SparseTensor):
    """
    对邻接矩阵 adj 进行对称归一化
    :param adj: torch_sparse.SparseTensor，表示图的邻接矩阵
    :return: torch_sparse.SparseTensor，归一化后的邻接矩阵
    """
    # 获取行索引和列索引
    row, col, value = adj.coo()

    # 计算度数：将每个节点的度数加 1（为了计算自环，也避免度数为 0 的节点带来的不稳定性）
    deg = torch_sparse.sum(adj, dim=1)  # 每行的元素和，即节点的度数

    # 计算 D^{-1/2} 的逆，避免除以零
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 防止无穷大的出现

    # 对 A 进行对称归一化： D^{-1/2} A D^{-1/2}
    value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

    # 构造归一化后的邻接矩阵
    norm_adj = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=adj.sparse_sizes())

    return norm_adj


def save_results(args, best_accuracy):
    global log_entries
    # 获取当前时间
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 文件名：dataset_name + best_accuracy + 时间戳
    filename = f"{args.dataset_name}/{timestamp}_acc:{best_accuracy:.4f}.txt"
    filepath = os.path.join('./results', filename)

    # 确保存储路径存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 将参数和结果写入文件
    with open(filepath, 'w') as f:
        f.write("Training Results:\n")
        f.write(f"Dataset Name: {args.dataset_name}\n")
        f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\nParameters:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\nTraining Logs:\n")
        f.write(log_entries)  # 写入训练日志

    print(f"Results saved to: {filepath}")


def printf(log_entry):
    """
    finally prints
    :param log_entry: 待打印的日志
    """
    global log_entries
    log_entries += str(log_entry) + "\n"  # 累积日志
    print(log_entry)


def printh(log_entry):
    """
    head prints
    :param log_entry: 待打印和追加的日志
    """
    global log_entries
    log_entries = str(log_entry) + "\n" + log_entries  # 累积日志
    print(log_entry)


def set_seed(seed):
    """设置随机种子，确保代码可重复运行"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 为了进一步确保确定性，可以设置以下环境变量
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path, dataset_name, acc=None):
    """
    保存模型的状态字典（state_dict）
    :param model: 当前的模型
    :param path: 保存路径
    """
    path = path+f'/{dataset_name}/{acc:.4f}.pth'
    torch.save(model.state_dict(), path)
    printf(f"模型已保存到 {path}")


def load_model(model, path):
    """
    加载模型的状态字典（state_dict）
    :param model: 当前模型实例
    :param path: 模型文件路径
    :return: 加载后的模型
    """
    # 加载保存的状态字典
    state_dict = torch.load(path)

    # 加载到模型中
    model.load_state_dict(state_dict)

    # 如果需要使用 GPU，记得加上设备参数
    model.eval()  # 设置为评估模式
    return model