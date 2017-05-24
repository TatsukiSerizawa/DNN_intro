# coding: utf-8
import numpy as np

#ReLUレイヤ
class Relu:
    def __init__(self):
        self.mask = None
    
    #順伝播
    def forward(self, x):
        self.mask = (x <= 0) #0以下ならTrue
        out = x.copy()
        out[self.mask] = 0 #インスタンス変数の値がTrueなら0を代入
        return out
    
    #逆伝播
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)
