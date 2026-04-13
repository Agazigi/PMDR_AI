import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils')
from utils import show_heatmaps, plot, Animator

# 训练集制作
n_train = 50 # 训练样本数
X_train, _ = torch.sort(torch.rand(n_train) * 5) # 这就是我们的输入特征
# torch.sort 对张量进行排序，返回排序后的张量和排序后的索引
def f(x):
    return 2 * torch.sin(x) + x**0.8
Y_train = f(X_train) + torch.normal(0.0, 0.5, (n_train,)) # 这就是我们的目标值

# 测试集制作
X_test = torch.arange(0, 5, 0.1) # 测试样本 0到5 0.1步长
Y_test = f(X_test)
n_test = len(X_test)

plt.plot(X_train.numpy(), Y_train.numpy(), 'o', alpha=0.5)
plt.plot(X_test.numpy(), Y_test.numpy(), 'k-')
plt.show()

def plot_kernel_reg(y_hat):
    plot(X_test, [Y_test, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    plt.plot(X_train, Y_train, 'o', alpha=0.5)

# 平均汇聚
# 缺点：忽略了输入 X
y_hat = torch.repeat_interleave(Y_train.mean(), n_test) # 重复n_test次 [50]
plot_kernel_reg(y_hat)
plt.show()


# 非参数注意力汇聚
# Nadaraya-Watson 核回归
# 相当于训练集中 x_i, y_i 是 key 和 value
# 我们的输入 x 是 query
# 注意力汇聚公式：
# f(x) = sum( alpha(x, x_i) * y_i ) 
# 这说明：
# 1. alpha(x, x_i) 是一个注意力权重，它表示 query x 和 key x_i 的相似度
# 2. y_i 是 value，它表示 key x_i 对应的 value

# 我们考虑使用高斯核来定义
# 最终 f(x) = sum( softmax( - (x - x_i) ** 2  / 2) * y_i )

# 从测试集 X_test 中重复 n_train 次，得到一个矩阵 X_repeat，
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = X_test.repeat_interleave(n_train).reshape((-1, n_train)) # [50, 50]

# 计算注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - X_train)**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, Y_train)
plot_kernel_reg(y_hat)
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
plt.show()

# 带参数注意力汇聚
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True)) # 权重参数 形状 [1]
    
    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = X_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = Y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(X_train, keys, values), Y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
    
# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = X_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = Y_train.repeat((n_test, 1))
y_hat = net(X_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
plt.show()
