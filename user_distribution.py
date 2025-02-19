import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

# 给定一组数字
# data = torch.load('Lastfm_user_social_count.pth')
# data1 = torch.load('Lastfm_user_item_count.pth')

data = torch.load('Ciao_user_social_count.pth')
data1 = torch.load('Ciao_user_item_count.pth')


data = data.cpu().numpy()
# data1 = data1

# 创建新的Y轴
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data, bins=25, alpha=0.6, color='r', edgecolor='black', linewidth=1.2, label='Histogram')  # 绘制直方图
bin_centers = 0.5 * (bins[:-1] + bins[1:])
ax.set_xticks(bin_centers)
ax.set_xticklabels([f'{int(center)}' for center in bin_centers], fontsize=6)

lines, labels = ax.get_legend_handles_labels()
# ax.legend(lines, labels, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Social Connections')
ax.set_ylabel('User Quantity')
plt.title('Ciao Social Connection Distribution')
plt.savefig('ciao_dist_social.pdf')


# a new figure
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data1, bins=25, alpha=0.6, color='r', edgecolor='black', linewidth=1.2, label='Histogram')  # 绘制直方图
bin_centers = 0.5 * (bins[:-1] + bins[1:])
ax.set_xticks(bin_centers)
ax.set_xticklabels([f'{int(center)}' for center in bin_centers], fontsize=6)

lines, labels = ax.get_legend_handles_labels()
# ax.legend(lines, labels, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Item Connections')
ax.set_ylabel('User Quantity')
plt.title('Ciao Interaction Distribution')
plt.savefig('ciao_dist_item.pdf')
plt.show()