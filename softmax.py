#coding:utf-8
import numpy as np

#ソフトマックス関数
def softmax(a):
    c = np.max(a) #配列内の最大値をcに代入
    exp_a = np.exp(a-c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    y_sum = np.sum(y)
    print(y)
    print(y_sum)