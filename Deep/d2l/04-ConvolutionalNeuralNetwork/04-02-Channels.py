import torch
import sys
sys.path.append('../utils')
from utils import *


# Multi Input Channels
def corr2d_multi_in(X, K):
    """多输入通道的互相关运算"""
    # X: [C_in, H, W]
    # K: [C_in, K, K]
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))


# Multi Output Channels
def corr2d_multi_in_out(X, K):
    """多输入通道的互相关运算，输出通道的互相关运算"""
    # X: [C_in, H, W]
    # K: [C_out, C_in, K, K]
    return torch.stack(
        [corr2d_multi_in(X, k) for k in K], dim=0
    )
K = torch.stack((K, K + 1, K + 2), 0)
print(corr2d_multi_in_out(X, K))


# 1x1 Conv
def corr2d_multi_in_out_1x1(X, K):
    # X: [C_in, H, W]
    # K: [C_out, C_in, 1, 1]
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w)) # [C_in, H * W]
    K = K.reshape((c_o, c_i)) # [C_out, C_in]
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X) # [C_out, H * W]
    return Y.reshape((c_o, h, w)) # [C_out, H, W]
X = torch.normal(0, 1, (3, 3, 3)) # [C_in, H, W]
K = torch.normal(0, 1, (2, 3, 1, 1)) # [C_out, C_in, 1, 1]

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(float(torch.abs(Y1 - Y2).sum()) < 1e-6)
