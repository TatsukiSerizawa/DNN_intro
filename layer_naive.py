# coding: utf-8

#乗算レイヤ(順伝播・逆伝播)
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x #インスタンス変数に値を保存
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

#加算レイヤ(順伝播・逆伝播)
class AddLayer:
    def __init__(self):
        pass #何も行わないという命令
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
