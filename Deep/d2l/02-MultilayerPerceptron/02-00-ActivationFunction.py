import torch
import sys
sys.path.append('../')
from utils.utils import *


if __name__ == '__main__':
    # ReLU
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    
    y.backward(torch.ones_like(x), retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
    
    # Sigmoid
    y = torch.sigmoid(x)
    plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
    
    # 清除以前的梯度
    x.grad.data.zero_()
    y.backward(torch.ones_like(x),retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
    
    # tanh
    y = torch.tanh(x)
    plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    
    # 清除以前的梯度
    x.grad.data.zero_()
    y.backward(torch.ones_like(x),retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
    plt.show()