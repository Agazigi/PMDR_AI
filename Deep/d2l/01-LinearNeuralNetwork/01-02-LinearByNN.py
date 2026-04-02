import torch 
import sys
sys.path.append('../')
from utils.utils import * 
import torch.nn as nn

# 真实数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

model = nn.Sequential(
    nn.Linear(2, 1) # 输入维度，输出维度
)

print('Model :', model)
model[0].weight.data.normal_(0, 0.01) # 正态分布
model[0].bias.data.fill_(0) # 全零

for params in model.parameters(): 
    print(params)
    
for module in model.modules(): 
    print(module)

lr = 0.03
num_epochs = 20


def Customer_Loss(y_hat, y, sigma=0.5):
    epsilon = torch.abs(y_hat - y)
    mask = epsilon > sigma
    loss = torch.zeros_like(epsilon)
    loss[mask] = epsilon[mask] - sigma / 2
    loss[~mask] = epsilon[~mask] ** 2 / (2 * sigma)
    return loss.sum()

loss = nn.MSELoss()
# loss = Customer_Loss
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
print('training on', device)


model.to(device)
for epoch in range(num_epochs):
    model.train()
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        output = model(X)
        l = loss(output, y)
        optimizer.zero_grad() # 首先清空原有参数的梯度
        l.backward() # 反向传播，计算梯度
        optimizer.step() # 更新参数
        
    model.eval()
    with torch.no_grad(): # 禁用梯度计算
        output = model(features.to(device))
        l = loss(output, labels.to(device))
    print(f'epoch {epoch + 1}, loss {l:f}')

model.to('cpu')
w = model[0].weight.data
print('w的估计误差: ', true_w - w.reshape(true_w.shape))
b = model[0].bias.data
print('b的估计误差: ', true_b - b)