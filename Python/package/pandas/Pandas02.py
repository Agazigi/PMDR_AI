import pandas as pd
import numpy as np
#
# # DataFrame
# # pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# # data：DataFrame 的数据部分，可以是字典、二维数组、Series、DataFrame 或其他可转换为 DataFrame 的对象。如果不提供此参数，则创建一个空的 DataFrame。
# # index：DataFrame 的行索引，用于标识每行数据。可以是列表、数组、索引对象等。如果不提供此参数，则创建一个默认的整数索引。
# # columns：DataFrame 的列索引，用于标识每列数据。可以是列表、数组、索引对象等。如果不提供此参数，则创建一个默认的整数索引。
# # dtype：指定 DataFrame 的数据类型。可以是 NumPy 的数据类型，例如 np.int64、np.float64 等。如果不提供此参数，则根据数据自动推断数据类型。
# # copy：是否复制数据。默认为 False，表示不复制数据。如果设置为 True，则复制输入的数据。
#
# # 使用字典创建DataFrame
data_2 = {
    'Text':['Hello','World','Pandas','Python','Data','Science','Machine','Learning','Deep','Learning','Deep','Learning','Deep']
    ,'Num':[1,2,3,4,5,6,7,8,9,10,11,12,13]
}
df_2 = pd.DataFrame(data_2)
df_2['Text'] = df_2['Text'].astype(str)
df_2['Num'] = df_2['Num'].astype('i4')
print(df_2)
#
# data = {
#     'name': ['Tom', 'Jack', 'Steve', 'Ricky'],
#     'age': [28, 34, 29, 42],
#     'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen']
# }
# df = pd.DataFrame(data)
# print(df)
# # 使用 np.array() 创建DataFrame
# data = np.array([
#     ['Hello',1]
#     ,['World',2]
#     ,['Pandas',3]
#     ,['Python',4]
#     ,['Data',5]
#     ,['Science',6]
#     ,['Machine',7]
#     ,['Learning',8]
#     ,['Deep',9]
#     ,['Learning',10]
#     ,['Deep',11]
#     ,['Learning',12]
#     ,['Deep',13]
# ])
# df = pd.DataFrame(data, columns=['Text','Num'])
# print(df)
# # 规定列的类型
# df['Text'] = df['Text'].astype(str)
# df['Num'] = df['Num'].astype('i4')
# '''
#     astype()
# '''
# # 对列进行操作
# df['Text'] = df['Text'].str.upper()
# df['Num'] = df['Num'].apply(lambda x: x*2)
# print(df)
# # 利用 Series 创建DataFrame
# s1 = pd.Series(['Alice', 'Bob', 'Charlie'])
# s2 = pd.Series([25, 30, 35])
# s3 = pd.Series(['New York', 'Los Angeles', 'Chicago'])
# df = pd.DataFrame({'Name': s1,
#                    'Age': s2,
#                    'City': s3})
# print(df)
# print(df.describe())
# print(df['Name'])
#
# # 直接放列表创建DataFrame
# df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                   columns=['Column1', 'Column2', 'Column3'])
# print(df)
# print(df.describe())
#
#
#
#
#
#
# data = [{'a': 1,
#          'b': 2},
#         {'a': 5,
#          'b': 10,
#          'c': 20}] # c 0 不存在
# df = pd.DataFrame(data)
# print (df)

# # 行索引，返回一行
# print(df_2.loc[0])
# print(df_2.loc[[0,1]])
#
#
#
#
# # 新增一行 concat
# new_row = {'Name': 'David', 'Age': 40, 'City': 'San Francisco'}
# # df = df._append(new_row, ignore_index=True) # ignore_index=True 是否忽略索引
# df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
# print(df)
#
# # 条件筛选
# print(df[df['Age'] > 30])
#
# df = df.drop(0) # 删除行
# df = df.drop(columns=['Name'])
# df = df.reset_index(drop=True) # 重置索引
# df = df.dropna() # 删除空值
# print(df)
#
# 纵向合并
df_1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']},
                   index=[0, 1, 2])
df_2 = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                    'D': ['D0', 'D1', 'D2']},
                   index=[0, 1, 2])
df = pd.concat([df_1, df_2], ignore_index=True, axis=0) # axis=0 是纵向合并, 默认是纵向合并，而横向合并是 axis=1
print(df)

# 横向合并
df_1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})
df_2 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'C': ['C0', 'C1', 'C2'],
                    'D': ['D0', 'D1', 'D2']})
df = pd.merge(df_1, df_2, on='A')
print(df)