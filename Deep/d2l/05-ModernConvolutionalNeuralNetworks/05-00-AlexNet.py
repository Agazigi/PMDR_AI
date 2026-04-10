import torch
import torch.nn as nn 
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt



class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 10)
        )
    def forward(self, X):
        return self.net(X)

if __name__ == '__main__':
    model = AlexNet()
    X = torch.randn(1, 1, 224, 224)
    for layer in model.net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params:,} parameters')
    
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    device = torch.device('cuda')
    train_gpu(model, train_iter, test_iter, num_epochs, lr, device)
    plt.show()
