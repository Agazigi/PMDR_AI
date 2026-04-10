import torch
import torch.nn as nn
import sys
sys.path.append('../utils')
from utils import *

# 卷积核/滤波器
class Conv2D(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size)) # (h, w)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

model = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
print(model.state_dict()) # [1, 1, 1, 2]


X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = model(X)
    l = (Y_hat - Y) ** 2
    model.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    model.weight.data[:] -= lr * model.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(model.weight)