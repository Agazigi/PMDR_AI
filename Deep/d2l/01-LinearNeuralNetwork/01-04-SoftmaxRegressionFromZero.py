import torch
import sys
sys.path.append('../')
from utils.utils import *


if __name__ == '__main__':
    batch_size = 256
    input_dim = 28 * 28
    num_classes = 10
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)

    W = torch.normal(0, 0.01, (input_dim, num_classes), requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)


    # W: [input_dim, num_classes]
    # b: [1, num_classes] b 通过广播，变成[batch_size, num_classes]

    # X: [batch_size, input_dim]

    # model: X @ W + b -> [batch_size, num_classes]


    def softmax(X):
        X_exp = torch.exp(X) # 全部都进行 exp [batch_size, num_classes]
        total = X_exp.sum(dim=1, keepdim=True) # [batch_size, 1] 按照行求和
        return X_exp / total # 广播

    def net(X):
        X = X.reshape(-1, input_dim)
        Z = torch.matmul(X, W) + b
        Y = softmax(Z)
        return Y


    def Cross_Entropy_Loss(y_hat, y):
        # y_hat: [batch_size, num_classes]
        # y: [batch_size, 1]
        return - torch.log(
            y_hat[range(len(y_hat)), y]
        )
        

    model = net
    loss = Cross_Entropy_Loss
    lr = 0.03
    epochs = 20
    def updater(batch_size):
        return sgd([W, b], lr, batch_size)


    train(model, train_dataloader, test_dataloader, loss, epochs, updater)
    predict(net, test_dataloader)
    plt.show()