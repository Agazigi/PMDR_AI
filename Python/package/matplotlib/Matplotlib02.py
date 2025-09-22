from matplotlib import pyplot as plt
import numpy as np
'''

plot()：用于绘制线图和散点图
scatter()：用于绘制散点图
bar()：用于绘制垂直条形图和水平条形图
hist()：用于绘制直方图
pie()：用于绘制饼图
imshow()：用于绘制图像
subplots()：用于创建子图

'''

xpoints = np.array([0, 6])
ypoints = np.array([0, 100])

plt.plot(xpoints, ypoints,'o')
# 'bo' 表示蓝色实心圆点
plt.show()

x = np.arange(0, 4 * np.pi, 0.1)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, x, z)
plt.show()