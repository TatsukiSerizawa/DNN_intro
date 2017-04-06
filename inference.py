#coding:utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from mnist import load_mnist
from softmax import softmax
from sigmoid_ReLU import sigmoid


def get_data():
    (x_train, t_train),(x_test, t_test) = \
        load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test,t_test

#学習済みの重みパラメータを読み込む
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 #バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size): #100枚ずつバッチとして取り出す
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch) #NNで分類処理
    p = np.argmax(y_batch, axis = 1) #最も確率の高い要素のインデックスを取得
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) #正解ラベルと比較して同じ（正解）なら+1
#    y = predict(network, x[i]) #NNで分類処理
#    if p == t[i]: #正解ラベルと比較して同じ（正解）なら+1
#        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) #認識精度
