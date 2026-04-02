import torch 
import torch.nn as nn 
import sys
sys.path.append('../')
from utils.utils import *

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.fc1 = nn.Linear(num_inputs, num_hiddens1)
        self.fc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        H1 = self.relu(self.fc1(X.reshape(-1, self.num_inputs)))
        if self.training == True:
            H1 = dropout_layer(H1, 0.5)
        H2 = self.relu(self.fc2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, 0.5)
        out = self.fc3(H2)
        return out

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    model = Model(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(model.parameters(), lr=lr)
    train(model, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show()
    
    
    
    
    
    dropout1 = 0.5
    dropout2 = 0.5
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10)
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show()