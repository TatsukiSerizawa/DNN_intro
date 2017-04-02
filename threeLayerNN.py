#coding: utf-8
import numpy as np
from functions import sigmoid #functionsファイルのsigmoid関数を呼ぶ

X = np.array([1.0, 0.5]) #入力
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #1重み
B1 = np.array([0.1, 0.2, 0.3]) #1バイアス

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) #2重み
B2 = np.array([0.1, 0.2]) #2バイアス

W3 = np.array([[0.1, 0.3], [0.2, 0.4]]) #3重み
B3 = np.array([0.1, 0.2]) #3バイアス

#第1層目の処理
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
#print(A1)
#print(Z1)

#第2層目の処理
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
#print(A2)
#print(Z2)

#第3層目の処理
def identity_function(x):
    return x
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) #またはY=A3
#print(A3)

print(Y) #結果出力