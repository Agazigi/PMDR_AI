import numpy as np

x = np.empty([3,2], dtype=np.int32) # 创建一个未初始化的数组
print(x)

y = np.zeros([3,3],dtype=np.int32) # 创建一个全零数组
print(y)

z = np.ones([3,2], dtype=np.int32) # 创建一个全1数组
print(z)

a = np.full([3,2], fill_value=5, dtype=np.int32) # 创建一个指定值数组
print(a)

b = np.eye(3, dtype=np.int32) # 创建一个单位矩阵
print(b)

c = np.random.randint(0, 10, size=[3,2], dtype=np.int32) # 创建一个随机整数组
print(c)

d = np.zeros_like(a, dtype=np.int32) # 创建一个与指定数组相同形状和类型的全零数组
print(d)

e = np.ones_like(a, dtype=np.int32) # 创建一个与指定数组相同形状和类型的全零数组
print(e)