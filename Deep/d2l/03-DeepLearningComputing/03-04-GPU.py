import torch
from torch import nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定程序可见的 GPU

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:0')) # cuda == cuda:0
print(torch.cuda.is_available())
print(torch.cuda.device_count())


x = torch.tensor([1, 2, 3])
print(x.device)
device = torch.device('cuda')
x = x.to(device)
print(x.device)
print(y := x.cuda(0)) # 在 GPU 0 上创建张量
print(y.device)


model = nn.Sequential(nn.Linear(3, 1))
model.to(device)
print(model)
print(model.state_dict())

Y = model(torch.rand(3, 3).to(device))
print(Y)
