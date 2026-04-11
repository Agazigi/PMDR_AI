import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (
        torch.zeros((batch_size, num_hiddens), device=device),
        torch.zeros((batch_size, num_hiddens), device=device)
    )

def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    H, C = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


class RNNModelScratch: 
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    

vocab_size, num_hiddens, device = len(vocab), 256, torch.device('cuda')
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
train_rnn(model, train_iter, vocab, lr, num_epochs, device)
plt.show()



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
    
num_inputs = vocab_size
model = RNN(len(vocab), num_hiddens)
model.rnn = nn.LSTM(num_inputs, num_hiddens)
model = model.to(device)
train_rnn(model, train_iter, vocab, lr, num_epochs, device)
plt.show()