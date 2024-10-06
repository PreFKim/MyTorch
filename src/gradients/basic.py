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
    
class Sum(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, dim, keepdim):  
        return x.data.sum(axis=dim, keepdims=keepdim)

    def backward(self, grad=1):
        node_x, dim, keepdim = self.saved_tensors
        if dim is not None:
            if isinstance(dim, tuple):
                dim = list(dim)
                for i in range(len(dim)):
                    if dim[i] < 0:
                        dim[i] = len(node_x.data.shape) + dim[i]
                dim = sorted(dim)
                
                dim_idx = 0

                new_shape = []
                for i, size in enumerate(node_x.shape):
                    if i == dim[dim_idx]:
                        new_shape.append(1)
                        dim_idx+=1
                    else:
                        new_shape.append(size)

                grad = grad.reshape(new_shape)
            else:
                grad = np.expand_dims(grad, dim)
        
        return np.broadcast_to(grad, node_x.data.shape)
            
class Mean(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, dim, keepdim):  
        return x.data.mean(axis=dim, keepdims=keepdim)

    def backward(self, grad=1):
        node_x, dim, keepdim = self.saved_tensors
        div = 1
        if dim is None:
            for size in node_x.shape:
                div *= size
        else:
            if isinstance(dim, tuple):
                dim = list(dim)
                for i in range(len(dim)):
                    if dim[i] < 0:
                        dim[i] = len(node_x.data.shape) + dim[i]
                dim = sorted(dim)
                
                dim_idx = 0
                new_shape = []

                for i, size in enumerate(node_x.shape):
                    if i == dim[dim_idx]:
                        div *= node_x.shape[i]
                        new_shape.append(1)
                        dim_idx+=1
                    else:
                        new_shape.append(size)
                grad = grad.reshape(new_shape)
            else :
                div = node_x.shape[dim]
                grad = np.expand_dims(grad, dim)
                
            
        return np.broadcast_to(grad, node_x.data.shape) / div