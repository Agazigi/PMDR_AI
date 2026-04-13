# 上一节给出了 Attention 机制（注意力汇聚） 的框架雏形：
# 1. 通过 queries 和 keys 计算注意力分数，例如 通过高斯核函数，就是一种评分函数。在 Transformer 中，我们使用点积作为评分函数。
# 2. 进行 softmax 运算，得到每个位置的权重（概率分布）。
# 3. 将权重与 values 相乘，得到加权和。

# 上一节中的高斯核处理的 标量
# 这一节的 加性注意力、点积注意力 可以处理矢量

# 这里要说一句，就是 num_steps 其实就是 seq_len

import torch
import torch.nn as nn
import sys
sys.path.append('../utils')
from utils import AdditiveAttention, DotProductAttention, show_heatmaps
import matplotlib.pyplot as plt



# 加性注意力机制
queries = torch.normal(0, 1, (8, 10, 24)) # [batch_size, num_steps, q_dim]
keys = torch.normal(0, 1, (8, 10, 24)) # [batch_size, num_steps, k_dim]
values = torch.arange(160, dtype=torch.float32).reshape(1, 10, 16).repeat(8, 1, 1) # [8, 10, 16] 
valid_lens = torch.tensor([10] * 8)
attention = AdditiveAttention(k_size=24, q_size=24, num_hiddens=32, dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens))
print(attention.attention_weights.shape) # [batch_size, num_steps, num_steps]
show_heatmaps(attention.attention_weights.reshape((2, 4, 10, 10)), xlabel='keys', ylabel='queries')
plt.show()


queries = torch.normal(0, 1, (8, 10, 24)) # [batch_size, seq_len, q_dim]
attention = DotProductAttention(dropout=0.5) # 点积注意力机制
attention.eval()
attention(queries, keys, values, valid_lens)
print(attention.attention_weights)
print(attention.attention_weights.shape)
show_heatmaps(attention.attention_weights.reshape((2, 4, 10, 10)), xlabel='keys', ylabel='queries')
plt.show()
