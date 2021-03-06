#coding: utf-8

import numpy as np

#損失関数(2乗和誤差)
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

'''
#損失関数(交差エントロピー誤差)
def cross_entropy_error(y, t):
    delta = 1e-7 #log0となってしまうことを防ぐ
    return -np.sum(t * np.log(y + delta))

#損失関数（[バッチ対応版]交差エントロピー誤差 教師データがラベルの時）
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
'''
#損失関数（[バッチ対応版]交差エントロピー誤差）
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7 #log0となってしまうことを防ぐ
    return -np.sum(t * np.log(y + delta)) / batch_size


#教師データ(2が正解とする)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#NNの出力(2の確率が最も高い場合(損失関数の値は小さくなる))
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
result1 = mean_squared_error(np.array(y), np.array(t))
result2 = cross_entropy_error(np.array(y), np.array(t))
print("2乗和誤差：", result1)
print("交差エントロピー誤差：", result2)

#7の確率が最も高い場合(損失関数の値は大きくなる)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
result1 = mean_squared_error(np.array(y), np.array(t))
result2 = cross_entropy_error(np.array(y), np.array(t))
print("2乗和誤差：", result1)
print("交差エントロピー誤差：", result2)
