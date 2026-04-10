from turtle import forward
import torch
import torch.nn as nn 
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, num_channels):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(self._conv_block(num_channels * i + in_channels, num_channels)) # 乘 i 是为了拼接前一层的输出
        self.net = nn.Sequential(*layers)
        
    def forward(self, X):
        for block in self.net:
            Y = block(X)
            X = torch.cat((X, Y), dim=1)
        return X
        
    def _conv_block(self, in_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        )


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        core_blocks = []
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            core_blocks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += growth_rate * num_convs
            if i != len(num_convs_in_dense_blocks) - 1:
                core_blocks.append(self._transition_block(num_channels, num_channels // 2))
                num_channels //= 2
            
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            *core_blocks,
            
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 10)
        )
    
    def forward(self, X):
        return self.net(X)

    def _transition_block(self, in_channels, num_channels): # 过渡层，为了减少通道数
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

if __name__ == '__main__':
    dense_block = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = dense_block(X)
    print(Y.shape)
    
    dense_net = DenseNet()

    n_params = sum(p.numel() for p in dense_net.parameters())
    print(f'{n_params:,} parameters')
        
    lr, num_epochs, batch_size = 0.1, 10, 64
    device = torch.device('cuda')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_gpu(dense_net, train_iter, test_iter, num_epochs, lr, device)
    plt.show()