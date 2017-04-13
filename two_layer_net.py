#coding: utf-8

import sys, os
sys.path.append(os.pardir) #親ディレクトリのファイルをインポートするための設定
from functions import *
from gradient2 import numerical_gradient

class TwoLayerNet:

    #初期化を行うメソッド(入力層ニューロン数，隠れ層ニューロン数，出力層ニューロン数)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #1層目の重み
        self.params['b1'] = np.zeros(hidden_size) #1層目のバイアス
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) #2層目の重み
        self.params['b2'] = np.zeros(output_size) #2層目のバイアス

        #認識（推論）を行うメソッド(xは画像データ)
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    # x:入力（画像）データ, t:教師データ（正解ラベル）

    #損失関数の値を求めるメソッド
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    #認識精度を求めるメソッド
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    #重みパラメータに対する勾配を求めるメソッド
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) #1層目の重みの勾配
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) #1層目のバイアスの勾配
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) #2層目の重みの勾配
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) #2層目のバイアスの勾配
        
        return grads

    #重みパラメータに対する勾配を求めるメソッド（numerical_gradientの高速版）        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward処理
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
