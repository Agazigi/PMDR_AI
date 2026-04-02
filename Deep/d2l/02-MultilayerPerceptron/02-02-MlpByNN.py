import torch
import torch.nn as nn
import torch.optim as optim 
import sys
sys.path.append('../')
from utils.utils import *

if __name__ == "__main__":
    epochs = 10
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    model.apply(init_weights)
    
    
    loss = nn.CrossEntropyLoss(reduction='none') # none 的原因是封装的函数会有对loss进行求和
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    train(model, train_iter, test_iter, loss, epochs, optimizer)
    predict(model, test_iter)
    plt.show()