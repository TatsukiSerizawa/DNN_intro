#coding: utf-8
import numpy as np

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])

print(W)

Y = np.dot(X, W) #内積の計算

print(Y)