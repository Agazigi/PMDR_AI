import numpy as np

a = [1, 2, 3]
b = np.asarray(a)
print(b)

x = [(1, 2, 3), (4, 5, 6)]
a = np.asarray(x)
print(a)

# 实现动态数组，将流的形式转化为 numpy 数组
s =  b'Hello World!' # byte string
k = np.frombuffer(s, dtype =  'S2') # 2 bytes per character
print (k)

# 将迭代对象建立为 numpy 数组
li = range(10)
it = iter(li)
c = np.fromiter(it, dtype = 'i4')
print(c)

# 生成 numpy 数组
d = np.arange(start=0, stop=10, step=2, dtype='i4')
print(d)

# 生成等差数列，是否包含 endpoint， 是否返回步长， 返回数据类型
e = np.linspace(start=1, stop=9, num=5, endpoint=True, retstep=True, dtype='i4')
print(e)

# 将数组转换为指定形状，当 reshape 为 False，才改变数组
f = np.linspace(start=1, stop=11, num=6, endpoint=True, retstep=False, dtype='i4').reshape(2, 3)
print(f)

# 创建一个等比数组
g = np.geomspace(start=1.0, stop=27, num=4)
print(g)

h = np.logspace(start=1.0, stop=10, num=10, base=2.0)
print(h)