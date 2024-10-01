from .grad import Grad
import numpy as np

class Get(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(node, idx):
        return node.data[idx]
    
    def backward(self, grad=1):
        node, idx = self.saved_tensors
        ret = np.zeros_like(node.data)
        ret[idx] = grad
        return ret


class Set(Grad):

    def __init__(self, prev_fn):
        super().__init__()
        self.prev_fn = prev_fn
        self.node_idx = None # For recurssive Set operation

    @staticmethod
    def forward(node, other, idx):
        node[idx].data = other.data
        return node.data
    
    def saved_for_backward(self, node, other, idx):
        if node.grad_fn is not None:
            self.saved_tensors = self.prev_fn.saved_tensors + (other, idx)
        else:
            self.saved_tensors = (node , other, idx)

        self.node_idx = len(self.prev_fn)
        
    
    def backward(self, grad=1):

        ret = []
        node, (other, idx) = self.saved_tensors[:-2], self.saved_tensors[-2:]
        
        if isinstance(grad, np.ndarray):
            grad1 = grad.copy()
        else:
            grad1 = np.ones_like(node[0].data)*grad
        grad1[idx] = 0
        ret.extend(self.prev_fn.backward(grad1))
            
        if isinstance(grad, np.ndarray):
            ret.append(grad[idx])
        else : 
            ret.append(np.ones_like(other.data)*grad)
        
        return ret
