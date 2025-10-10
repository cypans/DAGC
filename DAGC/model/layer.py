# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 13:29
@Auth: Panchuying
@File: layer.py
@IDE: PyCharm
@Theme:
"""
import torch
import torch_sparse
from torch import nn
from utils.utils import normalize_adj


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, in_feat_dim, out_feat_dim, k):
        super(GraphConv, self).__init__()

        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.node_feat_dim = in_feat_dim  # 节点特征维度
        self.out_feat_dim = out_feat_dim  # 输出特征维度
        self.k = k  # 卷积核阶数（即A^k的最大k值）

        # 定义每个输入通道的卷积核权重矩阵 Wk 和可学习的参数 alpha, beta, gamma 等
        self.W = nn.ParameterList([nn.ParameterList(nn.Parameter(torch.randn(in_feat_dim, out_feat_dim)) for _ in range(in_channels)) for _ in range(out_channels)])

        # 可学习的参数 alpha, beta, gamma, ... 对于每个阶数 A^k
        # 为每个输入通道和每个输出通道分别定义这些参数
        self.params = torch.nn.ParameterList([
            torch.nn.ParameterList([torch.nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in range(k+1)]) for _ in range(in_channels)])
            for _ in range(out_channels)])

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.uniform_(param, a=0.0, b=0.01)  # 在 [0, 0.1] 区间均匀采样
    def forward(self, x, edge_index):
        """
        前向传播
        :param x: 输入节点特征[in_channels,num_nodes,feat_dim]
        :param edge_index: 图的边
        :return: 输出特征
        """
        # 计算 A 的幂次 (I, A, A^2, ...)
        # 创建稀疏矩阵，指定边权重为 1
        value = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(x.size(1), x.size(1)))
        adj = normalize_adj(adj)  # 归一化 A

        # 计算 A 的不同阶数
        (row, col), _ = torch_sparse.eye(x.size(1))
        # 确保 row 和 col 的数据类型是 torch.long
        row = row.to(torch.long)
        col = col.to(torch.long)

        powers = [torch_sparse.SparseTensor(row=row, col=col,  sparse_sizes=(x.size(1), x.size(1))).to(x.device)]  # 创建稀疏矩阵并转移到设备 默认使用 COO 格式。如果没有指定 value，它会默认将所有非零元素的值设置为 None不能运算。
        for i in range(1, self.k + 1):
            powers.append(torch_sparse.matmul(powers[-1], adj))  # A^i = A^(i-1) * A     torch_sparse.matmul 函数只能用于 SparseTensor 类型的数据

        # 使用每个阶数和对应的参数计算卷积核
        out = torch.zeros(self.out_channels, x.size(1), self.out_feat_dim).to(x.device)  # 初始化输出

        for i in range(self.out_channels):  # 对每个输出通道
            for j in range(self.in_channels):  # 对每个输入通道
                for m in range(self.k+1):  # 对每个 A^m
                    # 计算卷积: A^m * Wk * x 并加权
                    out[i, :, :] += torch_sparse.matmul(powers[m], x[j, :, :]) @ self.W[i][j] * self.params[i][j][m]   #  聚合每个输出通道的所有输入通道
        # out [out_channels,num_nodes,feat_dim]
        # return self.bn(out.permute(1, 0, 2)).permute(1, 0, 2)
        # return self.bn(out)
        return out   # /(self.in_channels*self.k)