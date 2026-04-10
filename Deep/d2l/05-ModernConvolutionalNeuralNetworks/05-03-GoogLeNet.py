import torch
import torch.nn as nn 
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__()
        self.path_1_layer_1 = nn.Conv2d(in_channels, c1, kernel_size=1) # 1x1卷积
        
        self.path_2_layer_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1) # 1x1卷积
        self.path_2_layer_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) # 3x3卷积
        
        self.path_3_layer_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1) # 1x1卷积
        self.path_3_layer_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2) # 5x5卷积
        
        self.path_4_layer_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 最大池化层
        self.path_4_layer_2 = nn.Conv2d(in_channels, c4, kernel_size=1) # 1x1卷积
        
    def forward(self, X):
        path_1 = F.relu(self.path_1_layer_1(X))
        path_2 = F.relu(self.path_2_layer_2(F.relu(self.path_2_layer_1(X))))
        path_3 = F.relu(self.path_3_layer_2(F.relu(self.path_3_layer_1(X))))
        path_4 = F.relu(self.path_4_layer_2(self.path_4_layer_1(X)))
        return torch.cat((path_1, path_2, path_3, path_4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            
            nn.Flatten(),
            nn.Linear(1024, 10),
        )
    
    def forward(self, X):
        return self.net(X)
    
if __name__ == '__main__':
    model = GoogLeNet()
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params:,} parameters')

    lr, num_epochs, batch_size = 0.1, 10, 128
    device = torch.device('cuda')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_gpu(model, train_iter, test_iter, num_epochs, lr, device)
    plt.show()