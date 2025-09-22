import numpy as np

# 副本 与 视图
a = np.arange(6)
print(a)
print(id(a))

b = a.view()
c = a.copy()
b.shape = 3, 2
print(b)
print(id(b))
print(c)
print(id(c))

# IO
'''
e = np.array([1, 2, 3, 4, 5, 6])
np.save('out.npy', e)
print(np.load('out.npy'))

a = np.array([[1,2,3],[4,5,6]])
b = np.arange(0, 1.0, 0.1)
c = np.sin(b)
# c 使用了关键字参数 sin_array
np.savez("runoob.npz", a, b, sin_array = c)
r = np.load("runoob.npz")
print(r.files) # 查看各个数组名称
print(r["arr_0"]) # 数组 a
print(r["arr_1"]) # 数组 b
print(r["sin_array"]) # 数组 c
'''
e = np.array([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]])
np.savetxt("out.txt", e, delimiter=",", fmt="%d")
print(np.loadtxt("out.txt", delimiter=",", dtype=np.int32))
