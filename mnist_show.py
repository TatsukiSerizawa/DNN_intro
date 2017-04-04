#coding:utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image #PIL(Python Image Library)モジュールを使用

#訓練画像の1枚目を表示
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

#flatten=Trueで読み込んだ画像はNumPy配列として1次元で格納
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28) #28x28のサイズに再変形
print(img.shape)

img_show(img)
