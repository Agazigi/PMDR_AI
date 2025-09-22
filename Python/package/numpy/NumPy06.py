import numpy as np

# sin 函数
a = np.arange(0,181,30)
print(a)
print(list(np.sin(x*np.pi/180) for x in a))

# 矩阵操作
b = np.arange(12).reshape(3,4)
print(b)
print(b.sum())
print(b.sum(axis=0))
print(b.T) # 转置矩阵
print(b.max()) # 最大值
print(b.min()) # 最小值
print(b.argmax()) # 最大值索引
print(b.argmin()) # 最小值索引
print(b.cumsum()) # 累加
print(b.cumprod()) # 累乘
print(b.mean()) # 均值
print(b.std()) # 标准差
print(b.var()) # 方差
print(b.sort()) # 排序
print(b.sort(axis=1)) # 按行排序
print(b.sort(axis=0)) # 按列排序
print(b.argsort()) # 排序后索引
print(b.argsort(axis=1)) # 按行排序

c = np.identity(3) # 返回单位矩阵
print(c)

d = np.eye(3) # 返回单位矩阵
print(d)
e = np.eye(3,k=1)
print(e)

f = np.eye(3,4)
print(f)