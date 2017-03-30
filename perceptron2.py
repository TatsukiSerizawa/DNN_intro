# -*- coding: utf-8 -*-

import numpy as np

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

def AND(x1, x2):
    x = np.array([x1, x2]) #入力
    w = np.array([0.5, 0.5]) #重み
    b = -0.7 #バイアス
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    print('XOR[0,0] == ',XOR(0, 0))
    print('XOR[0,1] == ',XOR(0, 1))
    print('XOR[1,0] == ',XOR(1, 0))
    print('XOR[1,1] == ',XOR(1, 1))