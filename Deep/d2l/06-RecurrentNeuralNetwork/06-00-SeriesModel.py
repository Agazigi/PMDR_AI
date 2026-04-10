import torch
import torch.nn as nn
import sys
sys.path.append("../utils")
from utils import *
import matplotlib.pyplot as plt


# Series Data
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
plot(time, [x], 'time', 'x', xlim=(1, 1000), figsize=(6, 3))
plt.show()


# Feature Construction
delta = 4
features = torch.zeros((T - delta, delta)) # [996, 4]
for i in range(delta):
    features[:, i] = x[i: T - delta + i] # 第 i 个特征对应第 i 到 i + 996 个数据，这是直接处理的一整列
# label 就是 x[delta:]
labels = x[delta:].reshape((-1, 1)) # [996] -> [996, 1]
batch_size, n_train = 16, 600
train_iter = load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)


# Autoregressive Model 自回归的特点：每个时间步的预测值都是前 delta 个时间步的预测值的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        
model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
model.apply(init_weights)
loss = nn.MSELoss(reduction='none')
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {evaluate_loss(net, train_iter, loss):f}')

train(model, train_iter, loss, 10, 0.01)

one_step_preds = model(features) # 这里是用了原始数据作为输入特征，就是内插法：在现有观测值之间进行估计
plot(
    [time, time[delta:]], 
    [x.detach().numpy(), one_step_preds.detach().numpy()],
    'time', 'x', 
    legend=['data', '1-step preds'], 
    xlim=[1, 1000], 
    figsize=(6, 3)
)
plt.show()


# 外推法：对超出已知观测范围进行预测
multi_step_preds = torch.zeros(T) # [1000]
multi_step_preds[: n_train + delta] = x[: n_train + delta] # 初始化前 n_train + delta 个数据为训练集的数据
for i in range(n_train + delta, T):
    multi_step_preds[i] = model(multi_step_preds[i - delta: i].reshape((1, -1))) 

plot(
    [time, time[delta:], time[n_train + delta:]],
    [x.detach().numpy(), one_step_preds.detach().numpy(), multi_step_preds[n_train + delta:].detach().numpy()], 
    'time','x', 
    legend=['data', '1-step preds', 'multistep preds'],
    xlim=[1, 1000], 
    figsize=(6, 3)
)
plt.show()


max_steps = 64
features = torch.zeros((T - delta - max_steps + 1, delta + max_steps)) # [1000-4-64+1, 68]
# 列i（i<delta）是来自x的观测，其时间步从（i）到（i+T-delta-max_steps+1）
for i in range(delta):
    features[:, i] = x[i: i + T - delta - max_steps + 1]

# 列i（i>=delta）是来自（i-delta+1）步的预测，其时间步从（i）到（i+T-delta-max_steps+1）
for i in range(delta, delta + max_steps):
    features[:, i] = model(features[:, i - delta:i]).reshape(-1)

steps = (1, 4, 16, 64)
plot(
    [time[delta + i - 1: T - max_steps + i] for i in steps],
    [features[:, (delta + i - 1)].detach().numpy() for i in steps], 
    'time', 'x',
    legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
    figsize=(6, 3)
)
plt.show()

# Latent Autoregressive Model 潜变量自回归模型的特点：每个时间步的预测值都是前 delta 个时间步的预测值的函数，但是输入特征是潜变量