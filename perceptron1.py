# -*- coding: utf-8 -*-

# p25 パーセプトロンの簡単な実装
def AND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7 #2つの入力と閾値
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta: #ニューロンの発火
        return 1

# p27 重みとバイアスによるパーセプトロン
import numpy as np
def AND2(x1, x2):
    x = np.array([x1, x2]) #入力
    w = np.array([0.5, 0.5]) #重み
    b = -0.7 #バイアス
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) #重みとバイアスのみANDと異なる
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) #重みとバイアスのみ異なる
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    print('AND[0,0] ==',AND2(0, 0))
    print('AND[0,1] ==',AND2(0, 1))
    print('AND[1,0] ==',AND2(1, 0))
    print('AND[1,1] ==',AND2(1, 1))

    print('NAND[0,0] ==',NAND(0, 0))
    print('NAND[0,1] ==',NAND(0, 1))
    print('NAND[1,0] ==',NAND(1, 0))
    print('NAND[1,1] ==',NAND(1, 1))

    print('OR[0,0] ==',OR(0, 0))
    print('OR[0,1] ==',OR(0, 1))
    print('OR[1,0] ==',OR(1, 0))
    print('OR[1,1] ==',OR(1, 1))