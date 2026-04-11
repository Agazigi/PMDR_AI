import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../utils')
from utils import *

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
for X, Y in train_iter:
    print(X, Y)
    print(X.shape, Y.shape) # [32, 35], [32, 35]
    break

# One-Hot Encoding 需要创建一个 len(vocab) 长的 One-Hot Vector
print(len(vocab)) # 28 
# 理论来说，我们转换成 One-Hot 之后，维度应该是： [batch_size, num_steps, len(vocab)] 也就是 [32, 35, 28]
# 但实际上我们常用: [num_steps, batch_size, len(vocab)] 也就是 [35, 32, 28]
X = torch.randint(0, len(vocab), (32, 35)) # [32 ,35]
One_Hot_X = F.one_hot(X.T, num_classes=28)
print(One_Hot_X.shape) # [35, 32, 28]


class RNN:
    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens # len(vocab) = 28, num_hiddens = 512
        self.params = self._get_rnn_params(vocab_size, num_hiddens, device)
        self.embedding = nn.Embedding(vocab_size, vocab_size, device=device) # 能够将每个词转换为一个维度为 num_hiddens 的向量
        # 这里 Embedding 的维度是 vocab_size ，效果不如 512 等好一些
    def forward(self, batch_X, state, params):
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state # 这是包含 H_t-1 的状态 [batch_size, num_hiddens] -> [32, 512]
        outputs = []
        for X in batch_X:
            # 这里是按照时间步进行迭代 X_t
            # X: [batch_size, len(vocab)] -> [32, 28]
            
            # H_t = tanh(X_t @ W_xh + H_t-1 @ W_hh + b_h)
            # [batch_size, num_hiddens] = tanh( [batch_size, len(vocab)] @ [len(vocab), num_hiddens] + [batch_size, num_hiddens] @ [num_hiddens, num_hiddens] + [num_hiddens](广播) )
            # H_t: [batch_size, num_hiddens] -> [32, 512]
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            
            # Y_t = H_t @ W_hq + b_q
            # [batch_size, num_outputs] = [batch_size, num_hiddens] @ [num_hiddens, num_outputs] + [num_outputs](广播)
            # Y_t: [batch_size, num_outputs] -> [32, 28]
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,) # 将输出按照时间步拼接起来，并且返回 H_t
    
    def begin_state(self, batch_size, device):
        return self._init_rnn_state(batch_size, self.num_hiddens, device)
    
    def __call__(self, batch_X, state):
        # batch_X: [batch_size, num_steps]
        # state: [batch_size, num_hiddens]
        # batch_X = F.one_hot(batch_X.T, self.vocab_size).type(torch.float32) # Ont-Hot: [num_steps, batch_size, len(vocab)]
        
        # 其实这里也可以用 nn.Embedding 来实现 编码
        # 做成一个稠密向量
        batch_X = self.embedding(batch_X) # [batch_size, num_steps, num_hiddens]
        batch_X = batch_X.permute(1, 0, 2) # [num_steps, batch_size, num_hiddens]
        
        return self.forward(batch_X, state, self.params)
        
    def _init_rnn_state(self, batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device), )
        
    def _get_rnn_params(self, vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size # 28
    
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01
        
        W_xh = normal((num_inputs, num_hiddens)) # [28, 512]
        W_hh = normal((num_hiddens, num_hiddens)) # [512, 512]
        b_h = torch.zeros(num_hiddens, device=device) # [512]
         
        W_hq = normal((num_hiddens, num_outputs)) # [512, 28]
        b_q = torch.zeros(num_outputs, device=device) # [28]
        
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

num_hiddens = 512
device = torch.device('cuda')
model = RNN(len(vocab), num_hiddens, device)
state = model.begin_state(batch_size=batch_size, device=device) # [batch_size, num_hiddens] -> [32, 512]
X = torch.randint(0, len(vocab), (batch_size, num_steps), device=device) # [batch_size, num_steps] -> [32, 35]
Y, new_state = model(X, state)
print(Y.shape, new_state[0].shape) # [batch_size * num_steps, num_outputs] -> [32 * 35, 28] 相当于一个大的二维表，对应每一个输出
# 然后做交叉熵损失


print(predict_rnn('time traveller ', 10, model, vocab, device))
num_epochs, lr = 500, 1
train_rnn(model, train_iter, vocab, lr, num_epochs, device)
plt.show()


# OR:
# train_rnn(model, train_iter, vocab, lr, num_epochs, device, use_random_iter=True)
# plt.show()
