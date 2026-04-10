import torch
import torch.nn as nn 
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            self._nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), # 自适应平均池化层
            nn.Flatten()
        )
    def forward(self, X):
        return self.net(X)
    
    def _nin_block(self, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        
if __name__ == '__main__':
    model = NiN()
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params:,} parameters')

    lr, num_epochs, batch_size = 0.1, 10, 128
    device = torch.device('cuda')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    train_gpu(model, train_iter, test_iter, num_epochs, lr, device)
    plt.show()
