import torch
import torch.nn as nn
import sys
sys.path.append('../')
from utils.utils import *


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


if __name__ == '__main__':
    # DataLoader
    batch_size = 256
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)
    
    # Model
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]
    
    def net(X):
        X = X.reshape((-1, num_inputs)) # [Batch_size, 1, 28, 28] -> [Batch_size, 784]
        H = relu(X @ W1 + b1) # [Batch_size, 784] @ [784, 256] + [256] -> [Batch_size, 256]
        return (H @ W2 + b2) # [Batch_size, 256] @ [256, 10] + [10] -> [Batch_size, 10]
    
    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train(net, train_dataloader, test_dataloader, loss, num_epochs, updater)
    predict(net, test_dataloader)
    plt.show()