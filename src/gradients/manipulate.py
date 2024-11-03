from src.gradients.grad import Grad
import numpy as np

class Stack(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(nodes, dim):
        datas = [node.data for node in nodes]
        return np.stack(datas, dim)
    
    def backward(self, grad=1):
        nodes, dim = self.saved_tensors
        ret = [np.take(grad, indices=i, axis=dim) for i in range(len(nodes))]
        return ret

class Concat(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(nodes, dim):
        datas = [node.data for node in nodes]
        return np.concatenate(datas, dim)
    
    def backward(self, grad=1):
        nodes, dim = self.saved_tensors
        ret = [np.take(grad, indices=[i], axis=dim) for i in range(len(nodes))]
        return ret
    
class Reshape(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, shape):
        return np.reshape(x.data, shape)
    
    def backward(self, grad=1):
        node_x, shape = self.saved_tensors
        return np.reshape(grad, node_x.data.shape)

class Max(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return np.maximum(x.data, y.data)
    
    def backward(self, grad=1):
        node_x, node_y = self.saved_tensors
        
        grad_x = np.zeros_like(grad)
        grad_y = np.zeros_like(grad)

        x_idx = node_x>=node_y
        y_idx = node_x<node_y
        grad_x[x_idx] = grad[x_idx]
        grad_y[y_idx] = grad[y_idx]
        return grad_x, grad_y

class Min(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return np.minimum(x.data, y.data)
    
    def backward(self, grad=1):
        node_x, node_y = self.saved_tensors
        
        grad_x = np.zeros_like(grad)
        grad_y = np.zeros_like(grad)

        x_idx = node_x<=node_y
        y_idx = node_x>node_y
        grad_x[x_idx] = grad[x_idx]
        grad_y[y_idx] = grad[y_idx]
        return grad_x, grad_y