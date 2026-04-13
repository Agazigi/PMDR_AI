import torch
import sys
sys.path.append('../utils')
from utils import show_heatmaps
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 自主性注意力
# 非自主性注意力

# Query：自主性的提示，比如“我想要看一本书”。
# Key：非自主性的提示，比如“这是一本好书”。
# Query 和 Key 进行 注意力汇聚，得到匹配的权重。也就是注意力分数。

# Value：真正的本身信息。也就是“感官的输入”


attention_weights = torch.eye(10).reshape((1, 1, 10, 10)) #一行一列 10*10的矩阵
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
plt.show()

attention_weights = torch.rand((1, 1, 10, 10))
attention_weights = F.softmax(attention_weights, dim=-1)
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
plt.show()

attention_weights = attention_weights.reshape(10, 10)
print(attention_weights)
