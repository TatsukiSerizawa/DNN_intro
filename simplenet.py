#coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from functions import softmax, cross_entropy_error
from gradient2 import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) #ガウス分布で初期化
    
    #予測するためのメソッド
    def predict(self, x):
        return np.dot(x, self.W)
    
    #損失関数の値を求めるメソッド
    def loss(self, x, t): #xは入力データ，tは正解ラベル
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

def f(W):
    return net.loss(x, t)


if __name__ == "__main__":
    net = simpleNet()
    print(net.W) #重みパラメータ

    x = np.array([0.6, 0.9]) #入力値の設定
    p = net.predict(x) #ニューロン間の重みを用いた関数の計算
    print(p)
    print(np.argmax(p)) #最大値のインデックス

    t = np.array([0, 0, 1]) #正解ラベルの設定
    print(net.loss(x, t)) #損失関数から値を求める

    dW = numerical_gradient(f, net.W) #勾配を求める
    print(dW)
