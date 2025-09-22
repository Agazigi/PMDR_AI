import numpy as np
import pandas as pd


'''
    CSV 逗号分割文件
    以纯文本的格式存储数据
'''

# 读取
df = pd.read_csv('nba.csv')
print(df)
# print(df.to_string()) # 显示所有行

# 读取前边的 n 行，默认前 5 行
print(df.head())
# 读取后边的 n 行，默认后 5 行
print(df.tail())

print(df.info()) # 显示数据类型

# 写入
name = ['Tom', 'Nick', 'Kim']
age = [25, 30, 28]
city = ['Beijing', 'Shanghai', 'Guangzhou']
student = {
    'name': name,
    'age': age,
    'city': city
}
df = pd.DataFrame(student)
df.to_csv('student.csv')