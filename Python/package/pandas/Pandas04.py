import pandas as pd
import json
from glom import glom # 这个库可以使得嵌套数据读取更方便 通过 .

# JSON
df = pd.read_json('Site.json')
print(df)
print(df.to_string())

# 内嵌 JSON
with open('Sc.json','r') as json_file: # 读取文件
    data = json.loads(json_file.read()) # 将数据转换为字典
df = pd.json_normalize(data, record_path = ['students'], meta=['school_name', 'class']) # record_path 读取数据， meta 读取外层数据
# 设置 record_path 内嵌数据，默认为 None，读取数据
# 设置 meta 读取外层数据
print(df.to_string())

# 读取混合 JSON
with open('mix.json','r') as json_file:
    data = json.loads(json_file.read())

df = pd.json_normalize(data,
                       record_path = ['students'],
                       meta=['class',
                             ['info', 'president'],
                             ['info', 'contacts', 'tel']])
print(df.to_string())

# 读取嵌套 JSON 的属性
df = pd.read_json('grade.json')
print(df.to_string())
data = df['students'].apply(lambda row: glom(row, 'grade.math'))
# glom(row, 'grade.math') 是用来读取嵌套数据，row 是一行数据，grade.math 是要读取的数据
# data = data.tolist()
# 对 students 列的 grade.math 列进行读取
print(data)