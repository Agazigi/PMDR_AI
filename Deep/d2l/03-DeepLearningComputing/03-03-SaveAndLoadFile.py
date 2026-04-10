import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.arange(4)
torch.save(x, '../data/Save/Tensor_Save_File')

y = torch.load('../data/Save/Tensor_Save_File')
print(y)

z = torch.zeros(4, dtype=torch.float32)
torch.save([x, z], '../data/Save/Tensor_Save_Files')

x1, z1 = torch.load('../data/Save/Tensor_Save_Files')
print(x1)
print(z1)

MyDict = {
    'x': x,
    'y': y,
    'z': z,
}
torch.save(MyDict, '../data/Save/Tensor_Save_Dict')

MyDict1 = torch.load('../data/Save/Tensor_Save_Dict')
print(MyDict1)
print(MyDict1['x'])
print(MyDict1['y'])
print(MyDict1['z'])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))
    
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(Y)
print(net)


# 这存储的是一个类实例
torch.save(net, '../data/Save/MLP_Model_Save.pt')
Params = torch.load('../data/Save/MLP_Model_Save.pt', weights_only=False) # 现在不允许直接加载模型
print(Params)

print(net.state_dict()) # 这是 net 的参数字典，是一个 OrderedDict
torch.save(net.state_dict(), '../data/Save/MLP_Model_Save_Params.params')
Params = torch.load('../data/Save/MLP_Model_Save_Params.params')
print(Params)
print(Params.keys())

# 这时模型需要加载参数字典
net.load_state_dict(Params)
net.eval() # 切换到评估模式