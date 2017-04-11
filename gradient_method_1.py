#coding: utf-8
import numpy as np
from gradient1 import numerical_gradient

#勾配降下法
def gradient_descent(f, init_x, lr, step_num):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

#最適化したい関数
def function_2(x):
    return x[0]**2 + x[1]**2


if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0]) #初期値
    result = gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100)
    print("最小値:", result)

    #学習率が大きすぎる例
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(function_2, init_x = init_x, lr = 10.0, step_num = 100)
    print("学習率が大きすぎる最小値:", result)

    #学習率が小さすぎる例
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(function_2, init_x = init_x, lr = 1e-10, step_num = 100)
    print("学習率が小さすぎる最小値:", result)
