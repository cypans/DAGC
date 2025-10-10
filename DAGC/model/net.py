# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 13:31
@Auth: Panchuying
@File: net.py
@IDE: PyCharm
@Theme:
"""
from torch import nn
from model.layer import GraphConv
from torch.nn import functional as F


class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, node_feat_dim, hidden_feat_din, out_feat_dim, kernel_size, leaf):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_channels=in_channels, out_channels=5, in_feat_dim=node_feat_dim, out_feat_dim=hidden_feat_din, k=kernel_size[0])
        # self.conv2 = GraphConv(in_channels=3, out_channels=3, in_feat_dim=hidden_feat_din, out_feat_dim=hidden_feat_din, k=2)
        # self.conv3 = GraphConv(in_channels=8, out_channels=3, node_feat_dim=128, out_feat_dim=128, k=3)
        self.conv4 = GraphConv(in_channels=5, out_channels=out_channels, in_feat_dim=hidden_feat_din, out_feat_dim=int(leaf*out_feat_dim), k=kernel_size[1])  #dataset.num_classes
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        # self.dropout3 = nn.Dropout(p=0.5)
        self.lin1 = nn.Linear(node_feat_dim, hidden_feat_din)
        self.lin2 = nn.Linear(node_feat_dim, int(leaf*out_feat_dim))  # dataset.num_classes

    def reset_params(self):
        """递归重置所有子模块的参数,函数名不能与reset_parameters重名，不然会无限循环"""
        for module in self.modules():  # 遍历当前模块及所有子模块,包括linear，rnn那些.也包括本身，所以不能重名
            if hasattr(module, 'reset_parameters'):  # 检查模块是否有 reset_parameters 方法（linear，RNN都有）
                module.reset_parameters()

    def forward(self, x, edge_index):
        # x1 = x
        x = self.conv1(x.unsqueeze(0), edge_index) # + self.lin1(x.unsqueeze(0))
        x = self.dropout1(F.sigmoid(x))
        # x = self.conv2(x, edge_index)
        # x = self.dropout3(F.sigmoid(x))
        # x1 = x
        # x = self.conv2(x, edge_index)  # [out_channels, num_nodes, out_feat_dim]
        # x = F.sigmoid(x)
        # x = self.conv3(x, edge_index)  # [out_channels, num_nodes, out_feat_dim]
        # x = F.sigmoid(x)
        x = self.conv4(x, edge_index) # + self.lin2(x1)  # [out_channels, num_nodes, out_feat_dim]
        x = self.dropout2(x)

        return x.squeeze(0) #[:, :dataset.num_classes]