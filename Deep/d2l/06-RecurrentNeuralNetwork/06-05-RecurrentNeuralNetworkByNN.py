import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

class RNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(len(vocab), len(vocab)) # 将每个词转换为一个维度为 len(vocab) 的向量，并且转换成 [num_steps, batch_size, len(vocab)]
        # self.embedding = lambda X: F.one_hot(X.T.long(), self.vocab_size)
        self.rnn = nn.RNN(len(vocab), self.num_hiddens) # 默认是单向的、单层的 RNN
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            # 双向 RNN 有两个方向，所以 num_directions 是 2
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
        
    def _begin_state(self, batch_size, device):
        return torch.zeros((1, batch_size, self.num_hiddens), device=device)
    
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.RNN or nn.GRU 的 state 输出张量的形状是：(num_layers * num_directions, batch_size, num_hiddens)
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                device=device
            )
        else:
            # nn.LSTM 的 state 输出是两个张量，分别代表隐藏层和细胞状态
            return (
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                    device=device
                ),
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                    device=device
                )
            )
    
    def forward(self, X, state):
        X_embedding = self.embedding(X).to(torch.float32)
        if isinstance(self.embedding, nn.Embedding):
            X_embedding = X_embedding.permute(1, 0, 2)
        Y, state = self.rnn(X_embedding, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        # 首先将 Y reshape 成 [num_steps * batch_size, num_hiddens]
        # 然后 通过一个 linear 映射回 len(vocab) 维度
        return output, state

num_hiddens = 256
device = torch.device('cuda')
rnn = RNN(len(vocab), num_hiddens).to(device)
state = rnn._begin_state(batch_size=batch_size, device=device)
X = torch.rand(num_steps, batch_size, len(vocab)).to(device)
Y, new_state = rnn.rnn(X, state)
# nn.RNN() 的输入是：
# X: (num_steps, batch_size, vocab_size)
# state: (num_layers, batch_size, num_hiddens)
# 输出是：
# Y: (num_steps, batch_size, num_hiddens)
# state: (num_layers, batch_size, num_hiddens)
# 其中的 num_layers 代表 RNN 层数，默认是 1 层
print(f'【输入 X 的形状】: {X.shape}')
print(f'【输出 Y 的形状】: {Y.shape}')
print(f'【状态 new_state 的形状】: {new_state.shape}')


model = RNN(len(vocab), num_hiddens).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'【模型参数总数】: {n_params}')
predict_rnn('time traveller ', 10, model, vocab, device)
num_epochs, lr = 500, 1
train_rnn(model, train_iter, vocab, lr, num_epochs, device)
plt.show()