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

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = torch.device('cuda')
model = RNN(vocab_size, num_hiddens)
model.rnn = nn.LSTM(num_inputs, num_hiddens, num_layers) # 多层 LSTM
model = model.to(device)
num_epochs, lr = 500, 1
train_rnn(model, train_iter, vocab, lr, num_epochs, device)
plt.show()