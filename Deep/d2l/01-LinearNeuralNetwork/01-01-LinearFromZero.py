import torch
import torch.nn as nn
import random
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt

# 生成一个数据集 y = w_1 * x_1 + w_2 * x_2 + b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 可视化数据集
plt.figure(figsize=(4.5, 2.5))
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

plt.figure(figsize=(4.5, 2.5))
plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1)
plt.show()


# 随机采样
def data_iter(batch_size, features, labels):
    """随机采样"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# 模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# train
lr = 0.03
num_epochs = 3
model = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = model(X, w, b)
        l = loss(y_hat, y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(model(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')