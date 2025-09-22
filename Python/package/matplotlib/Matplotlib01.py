import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mtl
from matplotlib import font_manager

# 经典查看 matplotlib 版本
print(mtl.__version__)

# # 查看系统能直接调用的字体
# for font in font_manager.fontManager.ttflist:
#     # 查看字体名以及对应的字体文件名
#     print(font.name, '-', font.fname)

# MyFont = font_manager.FontProperties(fname="C:\Windows\Fonts\simfang.ttf")

x = np.arange(1,11)
y = 2 * x**2 + 5
plt.title("Matplotlib demo", fontsize=20) # 设置标题
plt.xlabel("x axis", fontsize=20) # 设置x轴标签
plt.ylabel("y axis", fontsize=20) # 设置y轴标签
plt.plot(x, y,'r') # 绘制折线图, 第一个参数是x轴数据，第二个参数是y轴数据，第三个参数是颜色
plt.show() # 显示图形
