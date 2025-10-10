import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 数据（根据原图大致估计Cora和CiteSeer的值，PubMed为假设值）
eta_values = [1, 2, 3, 4, 5]
cora_acc = [83.1, 83.6, 84.2, 84.4, 84.5]  # Cora 数据
citeseer_acc = [73.9, 74.2, 72.1, 74.0, 74.4]  # CiteSeer 数据
pubmed_acc = [80.0, 79.7, 79.9, 79.7, 80.0]  # PubMed 数据

# 创建图
plt.figure(figsize=(6, 4))
plt.plot(eta_values, cora_acc, 'b-o', label='Cora')
plt.plot(eta_values, citeseer_acc, 'r-^', label='CiteSeer')
plt.plot(eta_values, pubmed_acc, 'g-s', label='PubMed')

# 设置标题和标签
plt.title('(a) Effect of Parameter C')
plt.xlabel('C')
plt.ylabel('ACC (%)')
plt.legend()
plt.grid(True)

# 设置横坐标只显示整数 1, 2, 3, 4
plt.xticks(eta_values)

# 保存图像
plt.savefig('parameter_sensitivity_three_lines.png', dpi=300)


#______________________________________________________________________________________#
# 输入的5x5二维数据（已除以100，转换为小数形式）
ACC_data = np.array([
    [84.4, 84.4, 84.6, 84.4, 83.9],
    [83.8, 84.4, 84.3, 84.4, 84.2],
    [83.1, 84.6, 84.7, 84.4, 84.3],
    [83.2, 84.0, 84.5, 84.5, 84.0],
    [80.9, 82.7, 84.2, 84.0, 84.3]
]) / 100

# 创建λ₁和λ₂的范围（5个点）
lambda1 = np.linspace(1, 5, 5)  # λ₁ 范围
lambda2 = np.linspace(1, 5, 5)  # λ₂ 范围
Lambda1, Lambda2 = np.meshgrid(lambda1, lambda2)

# 使用输入的ACC数据
ACC = ACC_data

# 创建3D图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图
surface = ax.plot_surface(Lambda1, Lambda2, ACC, cmap='plasma', alpha=0.8)

# 绘制底部的等高线图（offset 设置为数据的最小值附近）
contour = ax.contourf(Lambda1, Lambda2, ACC, zdir='z', offset=0.0, cmap='plasma')

# 设置坐标轴标签
ax.set_xlabel('K₁')
ax.set_ylabel('K₂')
# ax.set_zlabel('accuracy')  # 数据是小数形式，标签改为 ACC
plt.xticks(lambda1)
plt.yticks(lambda2)
# 设置z轴范围（根据数据范围）
ax.set_zlim(0.0, 1.0)

# 添加颜色条
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

# 设置标题
plt.title("K")

# 保存图像
plt.savefig('sensitivity_analysis_3d_corrected.png', dpi=300)