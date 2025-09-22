import numpy as np

a = np.arange(5,61,5).reshape(3,2,2)
print(a)

for row in a:
    for col in row:
        print(col,end=',')
print('\n')

for row in a:
    print(row)

# 迭代器的遍历
for element in a.flat:
    print(element,end=',')
print('\n')

print(a.flatten()) # 返回 1 维数组
print(a.ravel()) # 返回 1 维数组