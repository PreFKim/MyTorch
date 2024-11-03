from src.gradients.grad import Grad
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
            grad * (node_y.data * node_x.data ** (node_y.data - 1)),
            grad * (node_x.data ** node_y.data * np.log(node_x.data))
            )

class Abs(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return np.abs(x.data)
    
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

class Log(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return np.log(np.clip(x.data, 1e-8, np.inf))

    def backward(self, grad=1):
        node_x, = self.saved_tensors
        return grad / node_x.data

class Sum(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, dim, keepdim):  
        return x.data.sum(axis=dim, keepdims=keepdim)

    def backward(self, grad=1):
        node_x, dim, keepdim = self.saved_tensors
        input_shape = node_x.shape

        grad = np.asarray(grad)

        if dim is None:
            if not keepdim:
                grad = grad.reshape(1)
            return np.broadcast_to(grad, input_shape)

        if isinstance(dim, (int, np.integer)):
            dim = [dim]
        elif isinstance(dim, tuple):
            dim = list(dim)

        for i in range(len(dim)):
            if dim[i] < 0:
                dim[i] = len(input_shape) + dim[i]
        dim = sorted(dim)

        if not keepdim:
            output_shape = [size for i, size in enumerate(input_shape) if i not in dim]
            grad = grad.reshape(output_shape)

            for d in dim:
                grad = np.expand_dims(grad, d)

        return np.broadcast_to(grad, input_shape)
    
class Mean(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, dim, keepdim):  
        return x.data.mean(axis=dim, keepdims=keepdim)

    def backward(self, grad=1):
        node_x, dim, keepdim = self.saved_tensors
        input_shape = node_x.shape

        grad = np.asarray(grad)

        if dim is None:
            div = np.prod(input_shape)
            if not keepdim:
                grad = grad.reshape(1)
            return np.broadcast_to(grad, input_shape) / div

        if isinstance(dim, (int, np.integer)):
            dim = [dim]
        elif isinstance(dim, tuple):
            dim = list(dim)

        for i in range(len(dim)):
            if dim[i] < 0:
                dim[i] = len(input_shape) + dim[i]
        dim = sorted(dim)
        
        div = np.prod([input_shape[i] for i in dim])

        if not keepdim:
            # Calculate output shape
            output_shape = [size for i, size in enumerate(input_shape) if i not in dim]
            grad = grad.reshape(output_shape)

        if not keepdim:
            for d in dim:
                grad = np.expand_dims(grad, d)

        return np.broadcast_to(grad, input_shape) / div
    