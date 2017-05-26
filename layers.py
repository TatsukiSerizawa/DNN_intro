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


#Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout): #Sigmoidレイヤの計算を入力と出力のみに省略し，計算を効率化したもの
        dx = dout * (1.0 - self.out) * self.out
        return dx

#Affineレイヤ（内積とバイアスの計算（バッチ対応版））
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        #重みバイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        #テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx
    
#Softmax-with-Lossレイヤ
#出力層のSoftmaxと損失関数の交差エントロピー誤差
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None #softmaaxの出力
        self.t = None #教師データ
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)
