import torch
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

batch_size, q_seq_len, valid_lens = 10, 32, torch.tensor([32] * 10)
X = torch.normal(0, 1, (batch_size, q_seq_len, num_hiddens))
print(attention(X, X, X, valid_lens).shape)
print(attention.attention.attention_weights.shape)
attention_map = attention.attention.attention_weights.reshape(batch_size, num_heads, q_seq_len, q_seq_len)
show_heatmaps(attention_map, xlabel='keys', ylabel='queries')
plt.show()


encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)', figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
show_heatmaps(P, xlabel='Column (encoding dimension)', ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
plt.show()
