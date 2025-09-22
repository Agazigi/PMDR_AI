from matplotlib import pyplot as plt
import numpy as np
import matplotlib.markers


y = np.array([1,3,4,5,8,9,6,1,3,4,5,2,4])
plt.plot(y, marker = 'o', ms = 20, mfc = 'r')
plt.title('Line Chart')
plt.xlabel('x position')
plt.ylabel('y position')

plt.grid()

plt.show()

'''

fmt = '[marker][line][color]'

"."	m00	点
","	m01	像素点
"o"	m02	实心圆
"v"	m03	下三角
"^"	m04	上三角
"<"	m05	左三角
">"	m06	右三角
"1"	m07	下三叉
"2"	m08	上三叉
"3"	m09	左三叉
"4"	m10	右三叉
"8"	m11	八角形
"s"	m12	正方形
"p"	m13	五边形
"P"	m23	加号（填充）
"*"	m14	星号
"h"	m15	六边形 1
"H"	m16	六边形 2
"+"	m17	加号
"x"	m18	乘号 x
"X"	m24	乘号 x (填充)
"D"	m19	菱形
"d"	m20	瘦菱形
"|"	m21	竖线
"_"	m22	横线
0 (TICKLEFT)	m25	左横线
1 (TICKRIGHT)	m26	右横线
2 (TICKUP)	m27	上竖线
3 (TICKDOWN)	m28	下竖线
4 (CARETLEFT)	m29	左箭头
5 (CARETRIGHT)	m30	右箭头
6 (CARETUP)	m31	上箭头
7 (CARETDOWN)	m32	下箭头
8 (CARETLEFTBASE)	m33	左箭头 (中间点为基准)
9 (CARETRIGHTBASE)	m34	右箭头 (中间点为基准)
10 (CARETUPBASE)	m35	上箭头 (中间点为基准)
11 (CARETDOWNBASE)	m36	下箭头 (中间点为基准)
"None", " " or ""	 	没有任何标记
'$...$'	m37	渲染指定的字符。例如 "$f$" 以字母 f 为标记。

'''

