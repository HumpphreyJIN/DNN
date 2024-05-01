import numpy as np
import matplotlib.pyplot as plt

# 加载权重矩阵
W1 = np.load(r'params/W1.npy')
W2 = np.load(r'params/W2.npy')
W3 = np.load(r'params/W3.npy')
weights = [W1, W2, W3]

# 设置子图布局
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    # 显示每个权重矩阵
    if i == 0:
        img = ax.imshow(weights[i], cmap='Greys', interpolation='nearest')
        ax.set_title(f'Weight Matrix {i+1}')
        ax.set_xlabel('Output Neurons')
        ax.set_ylabel('Input Neurons')
    else:
        img = ax.imshow(weights[i], cmap='Greys', interpolation='nearest')
        ax.set_title(f'Weight Matrix {i + 1}')
        ax.set_xlabel('Output Neurons')


# 在所有子图旁边添加一个共用的颜色条
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 调整颜色条的位置和大小
fig.colorbar(img, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # 调整布局，预留颜色条空间

fig.subplots_adjust(left=0.05, right=0.9, wspace=0.1)

plt.show()




