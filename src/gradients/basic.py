from .grad import Grad
import numpy as np


class Add(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x.data + y.data
    
    def backward(self, grad=1):
        return grad, grad

class Sub(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x.data - y.data
    
    def backward(self, grad=1):
        return grad, -grad


class Mul(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x.data * y.data
    
    def backward(self, grad=1):
        node_x, node_y = self.saved_tensors
        return (
            grad * node_y.data, 
            grad * node_x.data
            )

class Div(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x.data / y.data
    
    def backward(self, grad=1):
        node_x, node_y = self.saved_tensors
        return (
            grad * (1 / node_y.data), 
            grad * -(node_x.data / (node_y.data)**2)
            )   

class FloorDiv(Grad):
    @staticmethod
    def forward(x, y):
        return x.data // y.data

class Mod(Grad):
    @staticmethod
    def forward(x, y):
        return x.data % y.data

class Pow(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x.data ** y.data
    
    def backward(self, grad=1):
        node_x, node_y = self.saved_tensors
        return (
            grad * (node_x.data * node_y.data ** (node_x.data-1)),
            grad * (np.log(node_x.data) * node_x.data ** node_y.data)
            )

class Abs(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x.data[x.data<0] = -x.data[x.data<0]
        return x.data
    
    def backward(self, grad=1):
        node, = self.saved_tensors

        ret = grad.copy()
        ret[node.data<0] = -ret[node.data<0]

        return ret
    
class Neg(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return -x.data
    
    def backward(self, grad=1):
        return -grad

class MatMul(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):  
        return x.data @ y.data  

    def backward(self, grad=1):
        node_x, node_y = self.saved_tensors
        return (
            grad @ node_y.data.transpose(-1, -2),
            node_x.data.transpose(-1, -2) @ grad
            )