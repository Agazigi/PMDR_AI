import torch
import torch.nn as nn 
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, X):
        X_residual = X
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        if self.conv3:
            X_residual = self.conv3(X_residual)
        X += X_residual # 残差连接
        return F.relu(X)
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            *self._resnet_block(64, 64, 2, first_block=True),
            *self._resnet_block(64, 128, 2),
            *self._resnet_block(128, 256, 2),
            *self._resnet_block(256, 512, 2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
    
    def forward(self, X):
        return self.net(X)
    
    def _resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blocks = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blocks.append(Residual(input_channels, num_channels, use_1x1conv=True, stride=2))
            else:
                blocks.append(Residual(num_channels, num_channels))
        return blocks
    
    
if __name__ == '__main__':
    res = Residual(3,3)
    X = torch.rand(4, 3, 6, 6)
    Y = res(X)
    print(Y.shape)
    
    resnet18 = ResNet18()
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in resnet18.net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)
        
    n_params = sum(p.numel() for p in resnet18.parameters())
    print(f'{n_params:,} parameters')
        
    lr, num_epochs, batch_size = 0.1, 10, 64
    device = torch.device('cuda')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_gpu(resnet18, train_iter, test_iter, num_epochs, lr, device)
    plt.show()
