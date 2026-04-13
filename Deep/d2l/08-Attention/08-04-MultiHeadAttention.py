import torch
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

batch_size = 10
q_seq_len = 16
k_seq_len = 32
v_seq_len = 32
Q = torch.normal(0, 1, (batch_size, q_seq_len, num_hiddens))
K = torch.normal(0, 1, (batch_size, k_seq_len, num_hiddens))
V = torch.normal(0, 1, (batch_size, v_seq_len, num_hiddens))
valid_lens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 32]) # 键序列有效长度
# 注意， valid_lens 是针对于键序列的，而不是查询序列的
# Q 是正在解码的序列，没有 padding token
# 而 K 和 V 有 padding token，需要 valid_lens 来指定有效长度



print(attention(Q, K, V, valid_lens).shape) # [10, 16, 100] 得到的输出是 [batch_size, q_seq_len, hidden_dim]
print(attention.attention.attention_weights.shape) # [50, 16, 32] 
# 这里有 5 个头， 10 个批次，总共有 50 个注意力图
# 16 是 查询序列长度， 32 是 键序列长度
attention_map = attention.attention.attention_weights.reshape(batch_size, num_heads, q_seq_len, k_seq_len)
show_heatmaps(attention_map, xlabel='keys', ylabel='queries')
plt.show()