# -*- coding:utf-8 -*-
"""
@Time: 2024/12/20 13:35
@Auth: Panchuying
@File: run.py
@IDE: PyCharm
@Theme:
"""
import numpy as np

from utils.train_val import *
from utils.utils import save_results, printf, printh, save_model


def no_fold_train_and_test(model, loader, dataset, optimizer, criterion, args):
    # 训练模型并计算准确率
    max_acc = 0.0
    printed_flag = False  # 标志变量，初始为 False

    for epoch in range(args.epoch):
        try:
            if not printed_flag:
                printf("尝试全图加载")
            loss = train(model, loader, dataset, optimizer, criterion, args, "full")
            acc = test(model, loader, dataset, args, "full")
            if not printed_flag:
                printf("全图加载成功")
                printed_flag = True  # 设置标志，确保后续不再打印
        except RuntimeError as e:
            if "CUDA out of memory" or "insufficient resources when calling `cusparseSpGEMM_workEstimation" in str(e):
                if not printed_flag:
                    printf("显存溢出！尝试使用 ClusterLoader 加载数据...")
                loss = train(model, loader, dataset, optimizer, criterion, args, "cluster")
                acc = test(model, loader, dataset, args, "cluster")
                if not printed_flag:
                    printf("ClusterLoader 加载数据成功")
                    printed_flag = True  # 标记已切换到 cluster 加载
            else:
                raise e

        max_acc = max(max_acc, acc)
        # if acc == max_acc:
        #     save_model(model, "results", dataset_name=args.dataset_name, acc=max_acc)
        printf(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}%, Best Accuracy: {max_acc:.4f}%')
        # print(model.conv1.W[0][0])
        # print(model.conv1.params[0][0][0], model.conv1.params[0][0][1],model.conv1.params[0][0][2])
    return max_acc


def ten_fold_train_and_test(model, dataset, optimizer, criterion, args):
    # mask 分割 (形状 [nodes, 10])
    mask = dataset.get_idx_split(split_type="fixed_splits")  # 假设你已经加载了 mask，形状是 [nodes, 10]

    # 记录10次分割的测试准确率
    test_accuracies = []

    # 遍历10个分割的掩码
    for i in range(10):
        printf(f'------------------ 分割 {i + 1} -------------------------')

        # 提取单次分割掩码（形状 [nodes]）
        train_mask = mask["train_mask"][:, i]
        test_mask = mask["test_mask"][:, i]

        # 每次分割都重新初始化模型
        model.reset_params()  # 重置模型参数
        max_acc = 0.0
        # 训练
        for epoch in range(args.epoch):
            model.train()
            dataset.to(args.device)
            optimizer.zero_grad()
            out = model(dataset.graph["node_feat"], dataset.graph["edge_index"])
            loss = criterion(out[train_mask], dataset.label[train_mask])
            loss.backward()
            optimizer.step()

            # 测试
            model.eval()
            dataset.to(args.device)
            out = model(dataset.graph["node_feat"], dataset.graph["edge_index"])
            pred = out.argmax(dim=1)
            correct = pred[test_mask] == dataset.label[test_mask]
            acc = correct.sum().item() / correct.size(0) * 100
            max_acc = max(acc, max_acc)
            printf(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}%, Best Accuracy: {max_acc:.4f}%')

        test_accuracies.append(max_acc)
        # print(model.conv1.params[0][0][0], model.conv1.params[0][0][1], model.conv1.params[0][0][2])
        # print(model.conv4.params[0][0][0], model.conv4.params[0][0][1], model.conv4.params[0][0][2])
    # 计算10次分割的平均测试准确率
    avg_acc = sum(test_accuracies) / len(test_accuracies)
    # 计算均值和标准差
    acc_mean = np.mean(test_accuracies)
    acc_std = np.std(test_accuracies)
    printh(f"acc_list:{test_accuracies}")
    printh(f"Mean Accuracy: {acc_mean:.4f}")
    printh(f"Accuracy Standard Deviation: {acc_std:.4f}")
    return avg_acc


def run_one_time(model, loader, dataset, optimizer, criterion, args):
    printf(model)
    # if there are ten masks for each data,then we use the ten_fold
    if len(dataset.get_idx_split(split_type=args.split_type)["train_mask"].shape) == 2:
        acc = ten_fold_train_and_test(model, dataset, optimizer, criterion, args)
    else:
        acc = no_fold_train_and_test(model, loader, dataset, optimizer, criterion, args)
    # save_results(args, acc)
    return acc


def run_multiple_times(model, loader, dataset, optimizer, criterion, args, k):
    """
    运行 `run` 函数 k 次，计算每次的 acc，并返回均值和方差。

    参数:
        model: 模型
        loader: 数据加载器
        dataset: 数据集
        optimizer: 优化器
        criterion: 损失函数
        args: 参数对象
        k: 运行次数

    返回:
        acc_mean: k 次运行的 acc 均值
        acc_std: k 次运行的 acc 标准差
    """
    acc_list = []

    for i in range(k):
        print(f"Running iteration {i + 1}/{k}...")
        model.reset_params()
        # 运行单次模型并记录 acc
        if len(dataset.get_idx_split(split_type=args.split_type)["train_mask"].shape) == 2:
            acc = ten_fold_train_and_test(model, dataset, optimizer, criterion, args)
        else:
            acc = no_fold_train_and_test(model, loader, dataset, optimizer, criterion, args)
        acc_list.append(acc)
        print(f"Iteration {i + 1}, Accuracy: {acc:.4f}")

    # 计算均值和标准差
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)

    printh(f"\nFinal Results after {k} iterations:")
    printh(f'acc_list: {acc_list}')
    printh(f"Mean Accuracy: {acc_mean:.4f}")
    printh(f"Accuracy Standard Deviation: {acc_std:.4f}")

    save_results(args, 0000000)

    return acc_mean