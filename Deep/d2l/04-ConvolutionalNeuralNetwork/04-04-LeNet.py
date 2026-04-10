import torch
import torch.nn as nn
import sys
sys.path.append('../utils')
from utils import load_data_fashion_mnist, train_gpu
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            self.activation,
            nn.Linear(in_features=120, out_features=84),
            self.activation,
            nn.Linear(in_features=84, out_features=10)
        )
    
    def forward(self, X):
        X = self.pool1(self.activation(self.conv1(X)))
        X = self.pool2(self.activation(self.conv2(X)))
        X = X.flatten(1) # 从第2维开始展平，保留 Batch Size
        return self.fc(X)

if __name__ == '__main__':
    # model
    net = LeNet()
    print(net.state_dict().keys())

    # data
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # train
    lr, num_epochs, device = 0.9, 10, torch.device('cuda')
    train_gpu(net, train_iter, test_iter, num_epochs, lr, device)
    plt.show()
