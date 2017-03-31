#coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x1): #シグモイド関数
    return 1 / (1 + np.exp(-x1))

def relu(x2): #ReLU関数
    return np.maximum(0, x2)

x1 = np.arange(-5.0, 5.0, 0.1) #-5.0〜5.0まで0.1刻みでNumPy配列生成
y1 = sigmoid(x1)
plt.plot(x1, y1)

x2 = np.arange(-5.0, 5.0, 0.1)
y2 = relu(x2)
plt.plot(x2, y2)

plt.ylim(-0.1, 1.1) #y軸の範囲指定
plt.show()