import numpy as np

a = np.array([1,1,2,3,4,5,6,7,8,9])
# 去重
b = np.unique(a)
print(b)

# 布尔索引
c = a[a > 5]
print(c)

# 统计
d = np.count_nonzero(a > 5)
print(d)

# 广播
e = np.array([1,2,3])
f = np.array([[4,5,6]
              ,[7,8,9]
              ,[10,11,12]])
g = e + f
print(g)

# 排序
h = np.array([1,2,3,4,5,6,7,8,9])
i = np.sort(h)
print(i)

# 迭代数组
g = np.array([[1,2,3,4,5],[6,7,8,9,10]])
for x in np.nditer(g, op_flags=['readwrite']): # 默认情况下位只读
    x[...] = x * x # [...] 标识迭代到的元素索引
print(g)
