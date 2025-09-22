# tqdm 模块是 Python 的进度条显示库，主要分为两种运行模式。
# 1. 基于迭代对象运行
# 2. 手动进行更新

from time import sleep
from tqdm import tqdm
import random

# help(tqdm)

# 1.
l = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(l)
for i in pbar:
    pbar.set_description("Processing  " + i)
    sleep(0.5)

# 2.

with tqdm(total=10, desc="Processing") as pbar:
    for i in range(10):
        sleep(0.5)
        pbar.update(1)

# 3. 主要参数设置
epochs = 20
with tqdm(total=epochs, unit="epoch") as pbar:
    for epoch in range(epochs):
        pbar.set_description(f"[Training {epoch}/{epochs}]")
        pbar.set_postfix(loss=random.random())
        sleep(0.5)
        pbar.update(1)

# 4. 自定义进度条
with tqdm(total=10, bar_format="{postfix[0]}   value={postfix[1][value]}", postfix=["Batch", dict(value=0)]) as t:
    for i in range(10):
        sleep(0.1)
        t.postfix[1]["value"] = i / 2
        t.update()