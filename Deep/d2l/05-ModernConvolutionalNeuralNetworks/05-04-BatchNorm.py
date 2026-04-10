import torch
import torch.nn as nn 
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled(): # 测试时
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps) # 归一化
    else: # 训练时
        assert len(X.shape) in (2, 4) # 2表示全连接层，4表示卷积层
        if len(X.shape) == 2: # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else: # 卷积层
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape)) # 缩放参数
        self.beta = nn.Parameter(torch.zeros(shape)) # 平移参数
        self.moving_mean = torch.zeros(shape) # 移动均值
        self.moving_var = torch.zeros(shape) # 移动方差
        
    def forward(self, X):
        if self.moving_mean.device != X.device: # 如果X和参数不在同一个设备上，将参数复制到XSame
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y

if __name__ == '__main__':
    LeNet = net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        BatchNorm(6, num_dims=4), 
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(6, 16, kernel_size=5), 
        BatchNorm(16, num_dims=4), 
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), 
        
        nn.Flatten(),
        nn.Linear(16*4*4, 120), 
        BatchNorm(120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84), 
        BatchNorm(84, num_dims=2), 
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    
    lr, num_epochs, batch_size = 0.1, 10, 128
    device = torch.device('cuda')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_gpu(net, train_iter, test_iter, num_epochs, lr, device)
    plt.show()
    
    # Simple
    # net = nn.Sequential(
    #     nn.Conv2d(1, 6, kernel_size=5), 
    #     nn.BatchNorm2d(6),
    #     nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2),
        
    #     nn.Conv2d(6, 16, kernel_size=5),
        # nn.BatchNorm2d(16), 
    #     nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2), 
        
    #     nn.Flatten(),
    #     nn.Linear(256, 120), 
    #     nn.BatchNorm1d(120), 
    #     nn.Sigmoid(),
    #     nn.Linear(120, 84), 
    #     nn.BatchNorm1d(84),
    #     nn.Sigmoid(),
    #     nn.Linear(84, 10)
    # )