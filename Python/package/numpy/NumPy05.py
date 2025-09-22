import numpy as np

# 切片
a = np.arange(0, 10)
print(a)
b = slice(2, 7, 2)  # 从索引 2 开始，到索引 7 结束，步长为 2
print(a[b])
print(a[2:7:2])
print(a[::-1])

c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(c[1:])
print(c[:, 1])  # , 分隔要切的维数
print(c[1, 2])
print(c[:, 1:])

d = np.array(   [  [  [1, 2], [3, 4]  ],
                   [  [1, 2], [3, 4]  ],
                   [  [1, 2], [3, 4]  ] ])
print(d)
print(c[..., 1])