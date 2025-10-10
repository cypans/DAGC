# -*- coding:utf-8 -*-
"""
@Data: 2025-01-04 20:59 
@Auth: Panchuying
@File: dman_visization.py
@IDE: PyCharm
@Theme:
"""
from utils.args import args
from data_process.dataset import load_cluster_data
from model.net import GNN
from utils.utils import load_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


if __name__ == '__main__':
    loader, dataset = load_cluster_data(args.data_dir, args.dataset_name, args.split_type, args.num_parts,
                                        args.batch_size)
    dataset.to(args.device)
    model = GNN(in_channels=1, out_channels=1, node_feat_dim=dataset.graph['node_feat'].shape[-1], hidden_feat_din=args.hidden_feat_dim,
                out_feat_dim=dataset.num_classes, kernel_size=[args.kernel_size1, args.kernel_size2], leaf=args.leaf).to(args.device)
    model = load_model(model, args.model_path)
    out = model(dataset.graph["node_feat"], dataset.graph["edge_index"])
    print(out[dataset.graph["test_mask"]].shape)
    print(dataset.label[dataset.graph["test_mask"]].shape)

    # 提取模型输出和标签
    features = out[dataset.graph["test_mask"]].cpu().detach().numpy()  # 将输出移动到CPU并转为numpy数组
    labels = dataset.label[dataset.graph["test_mask"]].cpu().detach().numpy()

    # 使用 t-SNE 进行降维，降到2D空间
    tsne = TSNE(n_components=2, random_state=42)  # 使用2维空间
    features_2d = tsne.fit_transform(features)

    # 绘制 t-SNE 可视化图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='plasma', s=15)
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of DMNA model output")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

    # 保存图像到文件
    plt.savefig(f'./results/{args.dataset_name}/tsne_visualization.png', dpi=300)  # 保存为 PNG 格式，300 dpi 的分辨率
    plt.close()  # 关闭图形以释放内存