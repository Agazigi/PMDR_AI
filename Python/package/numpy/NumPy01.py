import numpy as np

# numpy 的多维数组
a = np.array([1, 2, 3])
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 的数组
print(b)

c = np.array([1, 2, 3], dtype=np.float64)
print(c)

d = np.array([   [1, 2, 3],
                        [4, 5, 6]   ], dtype=np.float64)
print(d)

e = np.array([1, 2, 3], dtype=complex)
print(e)

f = np.array(   [   [1+1j, 2+2j, 3+3j],
                    [4+4j, 5+5j, 6+6j]  ]     ) # 复数数组
print(f)

g = np.zeros((3, 3)) # 全零数组
print(g)

h = np.ones((3, 3)) # 全一数组
print(h)

i = np.eye(3) # 单位矩阵
print(i)

j = np.random.rand(3, 3) # n x m 随机数组
print(j)

k = np.random.randn(3, 3) # n x m 正态分布的随机数组
print(k)

l = np.random.randint(0, 10, (3, 3)) # n x m 随机整数数组
print(l)

m = np.arange(0, 10, 2) # 等差数列 [0,10) d = 2
print(m)

n = np.linspace(0, 1, 5) # 等差数列 [0,1) d = 5
print(n)

o = np.full((3, 3), 7) # 3 x 3 数组，每一项都为 7
print(o)

p = np.array([[1, 2], [3, 4], [5, 6]])
print(p)


# numpy 数组的属性
print(p.shape) # 数组的大小 (3, 2)
print(p.size) # 数组的总元素个数
print(p.ndim) # 数组的维度
print(p.dtype) # 数组元素的数据类型
print(p.itemsize) # 数组元素的字节数
print(p.nbytes) # 数组的总字节数
print(p.flags) # 数组的标志
print(p.strides) # 数组各维度上遍历下一元素的步长
print(p.base) # 数组的基数组
print(p.data) # 数组的数据地址
print(p.flags.writeable) # 数组的可写性
print(p.flags.owndata) # 数组是否拥有自身的数据缓冲区
print(p.flags.c_contiguous) # 数组的存储顺序为 C
print(p.flags.f_contiguous) # 数组的存储顺序为 Fortran
print(p.flags.fnc) # # 数组是否连续存储
