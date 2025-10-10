# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 15:45
@Auth: Panchuying
@File: train_val.py
@IDE: PyCharm
@Theme:
"""
import torch
from tqdm import tqdm


def train(model, loader, dataset, optimizer, criterion, args, train_type="cluster"):
    """
    :param train_type: 训练模式：'full' or 'cluster'
    :return
    """
    model.train()
    if train_type == "cluster":
        total_loss = 0.0
        if args.dataset_name == "ogbn-products" or len(loader) >= 50:
            for subgraph_data in tqdm(loader):
                if not torch.sum(subgraph_data.train_mask) == 0:
                    subgraph_data = subgraph_data.to(args.device)
                    optimizer.zero_grad()
                    out = model(subgraph_data.x, subgraph_data.edge_index)
                    loss = criterion(out[subgraph_data.train_mask], subgraph_data.y[subgraph_data.train_mask])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        else:
            for subgraph_data in loader:
                if not torch.sum(subgraph_data.train_mask) == 0:
                    subgraph_data = subgraph_data.to(args.device)
                    optimizer.zero_grad()
                    out = model(subgraph_data.x, subgraph_data.edge_index)
                    loss = criterion(out[subgraph_data.train_mask], subgraph_data.y[subgraph_data.train_mask])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        return total_loss / len(loader)

    elif train_type == "full":
        model.train()
        dataset.to(args.device)
        optimizer.zero_grad()
        out = model(dataset.graph["node_feat"], dataset.graph["edge_index"])
        loss = criterion(out[dataset.graph["train_mask"]], dataset.label[dataset.graph["train_mask"]])
        loss.backward()
        optimizer.step()
        return loss.item()


# 准确率计算
def test(model, loader, dataset, args, test_type="cluster"):
    """
    :param test_type: str "full" or "cluster"
    :return:
    """
    model.eval()
    if test_type == "cluster":
        total_correct = 0.0
        total_count = 0.0
        if len(loader)>=50:
            for subgraph_data in tqdm(loader):
                subgraph_data = subgraph_data.to(args.device)
                out = model(subgraph_data.x, subgraph_data.edge_index)
                pred = out.argmax(dim=1)
                correct = pred[subgraph_data.test_mask] == subgraph_data.y[subgraph_data.test_mask]
                total_correct += correct.sum().item()
                total_count += correct.size(0)
            acc = total_correct / total_count

            return acc * 100
        else:
            for subgraph_data in loader:
                subgraph_data = subgraph_data.to(args.device)
                out = model(subgraph_data.x, subgraph_data.edge_index)
                pred = out.argmax(dim=1)
                correct = pred[subgraph_data.test_mask] == subgraph_data.y[subgraph_data.test_mask]
                total_correct += correct.sum().item()
                total_count += correct.size(0)
            acc = total_correct / total_count

            return acc * 100

    elif test_type == "full":
        model.eval()
        dataset.to(args.device)
        out = model(dataset.graph["node_feat"], dataset.graph["edge_index"])
        pred = out.argmax(dim=1)
        correct = pred[dataset.graph["test_mask"]] == dataset.label[dataset.graph["test_mask"]]
        acc = correct.sum().item() / correct.size(0)
        return acc * 100