import numpy as np

# 线性代数
a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([1,2,3,4,5,6,7,8,9])
print(np.dot(a,b)) # 点积

c = np.array([1+2j,2+3j])
d = np.array([2+1j,3+4j])
print(np.inner(c,d)) # 内积
print(np.dot(c,d)) # 点积
print(np.vdot(c,d)) # 虚积

e = np.array([[1,2],[3,4]],dtype='i4')
f = np.array([[11,12],[13,14]])
print(np.dot(e,f)) # 矩阵乘法
print(np.inner(e,f)) # 内积
print(np.matmul(e,f)) # 矩阵乘法

print(np.linalg.det(e)) # 矩阵行列式， 返回一个浮点数

g = np.linalg.inv(e)
print(g) # 矩阵的逆，返回一个数组
print(np.dot(g,e))

# 求解线性方程组
h = np.array([[1,1,1],[0,2,5],[2,5,-1]])
i = np.array([[6],[-4],[27]])
print(np.linalg.solve(h,i)) # solve函数，将增广矩阵的 A ， b 放入，求解方程组


