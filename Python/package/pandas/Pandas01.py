import pandas as pd
import numpy as np

'''
    Pandas 中包含两种数据结构
    Series: 类似于一维数组，一组数据和与之相关的数据标签（索引）的组合
    DataFrame: 类似于一个表格，有一组有序的列，每列可以是不同的数据类型
'''

# print(pd.__version__)

# Test = {
#     'name': ['Tom', 'Jack', 'Steve', 'Ricky'],
#     'age': [28, 34, 29, 42],
#     'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'] # 注意格式
# }
#
# data = pd.DataFrame(Test)
# print(data)

# a = np.array([1, 2, 3, 4, 5])
# b = pd.Series(data = a, name = 'data', index = ['a', 'b', 'c', 'd', 'e'], dtype = float)
# print(b)
#
# # 我们也可以存储 key-value 的数据结构，例如字典
# data = {
#     1: 'Jack',
#     2: 'Tom',
#     3: 'Steve',
#     4: 'Ricky'
# }
# b = pd.Series(data, index=[1,2,3,4])
# print(b)





# pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)
# data：Series 的数据部分，可以是列表、数组、字典、标量值等。如果不提供此参数，则创建一个空的 Series。
# index：Series 的索引部分，用于对数据进行标记。可以是列表、数组、索引对象等。如果不提供此参数，则创建一个默认的整数索引。
# dtype：指定 Series 的数据类型。可以是 NumPy 的数据类型，例如 np.int64、np.float64 等。如果不提供此参数，则根据数据自动推断数据类型。
# name：Series 的名称，用于标识 Series 对象。如果提供了此参数，则创建的 Series 对象将具有指定的名称。
# copy：是否复制数据。默认为 False，表示不复制数据。如果设置为 True，则复制输入的数据。
# fastpath：是否启用快速路径。默认为 False。启用快速路径可能会在某些情况下提高性能。

# c = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
# for index, value in c.items(): # .item() 返回元组列表 ，即 (index, value) ， 直接 c 就是元素
#     print(f"{index} is {value}")
# for i in c.items():
#     print(f"{i[0]} is {i[1]}")

# del c[0]
# print(c)
#
# # 全部乘以 2
# c = c * 2
# print(c)
#
# filter_c = c[c > 5]
# print(filter_c)
#
# status = c.describe()
# print(status)