import torch 
import torch.nn as nn 
from torch.utils import data
import torchvision
from torchvision import transforms
import sys
sys.path.append('../')
from utils.utils import *

trans = transforms.Compose([
    transforms.ToTensor() # 转换为张量
])


# [Batch_size, 1, H, W]

if __name__ == '__main__':
    mnist_train_data = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)


    print(len(mnist_train_data), len(mnist_test_data))
    print(mnist_train_data[0][0].shape) # [1, H, W]


    X, y = next(iter(data.DataLoader(mnist_train_data, batch_size=36)))
    show_images(X.reshape(36, 28, 28), 6, 6, titles=get_fashion_mnist_labels(y))
    plt.show()


    batch_size = 256
    num_workers = 4
    train_dataloader = data.DataLoader(mnist_train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = data.DataLoader(mnist_test_data, batch_size=batch_size, num_workers=num_workers)

    for X, y in train_dataloader:
        print(X.shape, y.shape)
        break