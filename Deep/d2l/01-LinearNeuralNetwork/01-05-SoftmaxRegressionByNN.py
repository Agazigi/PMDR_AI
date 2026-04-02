import torch
import torch.nn as nn
import sys
sys.path.append('../')
from utils.utils import *


if __name__ == '__main__':
    batch_size = 256
    epochs = 20
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)
    model = nn.Sequential(
        nn.Flatten(), # 这个会将输入的二维数据展平成一维数据
        nn.Linear(28 * 28, 10)
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            
    model.apply(init_weights) # 对model中的所有Linear层进行初始化
    loss = nn.CrossEntropyLoss(reduction='none') # Softmax + 交叉熵 混合在了一起
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train(model, train_dataloader, test_dataloader, loss, epochs, optimizer)
    plt.show()