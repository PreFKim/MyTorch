from src.gradients.grad import Function, unbroadcast
import numpy as np

class Get(Function):

    @staticmethod
    def forward(ctx, node_x, idx):
        ctx.saved_for_backward(node_x)
        ctx.idx = idx
        return node_x.data[idx]
    
    @staticmethod
    def backward(ctx, grad=1):
        node_x, = ctx.saved_tensors
        idx = ctx.idx

        ret = np.zeros_like(node_x.data)
        ret[idx] = grad
        return ret


class Set(Function):
    @staticmethod
    def forward(ctx, node, other, idx):
        ctx.idx = idx
        node.data[idx] = other.data
        return node.data   

    @staticmethod
    def backward(ctx, grad=1):
        idx = ctx.idx

        grad1 = grad.copy()
        grad1[idx] = 0

        grad2 = grad[idx]
        return grad1, grad2
