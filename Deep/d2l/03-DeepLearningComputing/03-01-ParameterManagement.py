import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8), 
    nn.ReLU(), 
    nn.Linear(8, 1)
)
X = torch.rand(size=(2, 4))
print(net(X))


print('net: ', net)
print()
print('net state dict: ', net.state_dict())
print()

for i in range(len(net)):
    print(f'=== Layer {i} ===')
    print(net[i])
    print(net[i].state_dict())
    for name, param in net[i].named_parameters():
        print(name, param.size(), param.numel())
    print()
    
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].bias.grad)
print()


for name, param in net.named_parameters():
    print(name, param.size(), param.numel())
    
    
    
    
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print()
print(rgnet)


# 稠密层 

# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8), 
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared, 
    nn.ReLU(),
    nn.Linear(8, 1)
)
print()
print(net)
print(net(X))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])